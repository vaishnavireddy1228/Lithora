
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# 1. CONFIGURATION
# ==========================================

tf.keras.utils.set_random_seed(42)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20

MANIFEST_PATH = Path("dataset/image_manifest.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = "models/lithora_model.keras"

# ==========================================
# 2. LOAD AND VALIDATE DATASET MANIFEST
# ==========================================

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

df = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "image_path", "patient_id", "label", "group"
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Manifest must contain these columns: {required_columns}"
    )

print("\nImages per split:")
print(df.groupby(["group", "label"]).size().to_string())

# Label encoding: Non-Stone = 0, Stone = 1
label_map = {
    "Non-Stone": 0,
    "Stone": 1
}

df["label_id"] = df["label"].map(label_map)

if df["label_id"].isna().any():
    raise ValueError("Unexpected label found in manifest.")

if df[["image_path", "patient_id", "group"]].isna().any().any():
    raise ValueError("Manifest contains missing required values.")

# Check that each patient appears in only one split
patient_split_counts = df.groupby("patient_id")["group"].nunique()

if (patient_split_counts > 1).any():
    raise ValueError(
        "Patient leakage detected: some patients appear in multiple splits."
    )

# Check image paths
missing_paths = [
    path for path in df["image_path"]
    if not Path(path).is_file()
]

if missing_paths:
    raise FileNotFoundError(
        f"{len(missing_paths)} images are missing. "
        f"Example: {missing_paths[0]}"
    )

# ==========================================
# 3. CREATE TRAIN, VALIDATION AND TEST SPLITS
# ==========================================

train_df = df[df["group"] == "train"].copy()
val_df = df[df["group"] == "validation"].copy()
test_df = df[df["group"] == "test"].copy()

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError("Training, validation or test split is empty.")

print("\nSplit sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    # Dataset contains JPEG images.
    image = tf.io.decode_jpeg(
        image_bytes,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # Convert pixels to float values between 0 and 1.
    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.float32)


def make_dataset(frame, training=False):
    paths = frame["image_path"].to_numpy()

    labels = frame["label_id"].to_numpy(
        dtype="float32"
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=2
    )

    dataset = dataset.batch(BATCH_SIZE)

    return dataset.prefetch(1)


train_ds = make_dataset(
    train_df,
    training=True
)

val_ds = make_dataset(val_df)

test_ds = make_dataset(test_df)

# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

# Applied during training only.
# Keep transformations mild to preserve anatomy.

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(0.03, 0.03),
], name="data_augmentation")

# ==========================================
# 6. BUILD IMPROVED CNN
# ==========================================

inputs = keras.Input(
    shape=(128, 128, 1)
)

x = data_augmentation(inputs)

# Convolution block 1
x = layers.Conv2D(
    32,
    kernel_size=3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)

# Convolution block 2
x = layers.Conv2D(
    64,
    kernel_size=3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)

# Convolution block 3
x = layers.Conv2D(
    128,
    kernel_size=3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)

# Classification head
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    64,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="Lithora_CNN_V2"
)

# ==========================================
# 7. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0003
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# ==========================================
# 8. TRAINING CALLBACKS
# ==========================================

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nStarting Lithora CNN V2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],
    class_weight={
        0: 1.0,  # Non-Stone
        1: 1.5   # Stone: give missed-class errors more weight
    }
)

# ==========================================
# 10. LOAD BEST CHECKPOINT
# ==========================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        "Best model checkpoint was not saved."
    )

best_model = keras.models.load_model(
    MODEL_PATH
)

# ==========================================
# 11. FINAL TEST EVALUATION
# ==========================================

# The test set is evaluated after model selection.
print("\nEvaluating the saved model on test data...")

test_results = best_model.evaluate(
    test_ds,
    return_dict=True,
    verbose=1
)

print("\nBest validation loss:")
print(min(history.history["val_loss"]))

print("\nBest validation accuracy:")
print(max(history.history["val_accuracy"]))

print("\nFinal training accuracy:")
print(history.history["accuracy"][-1])

print("\nFinal validation accuracy:")
print(history.history["val_accuracy"][-1])

print("\nLithora CNN V2 test results:")

for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

print("\nOriginal model preserved:")
print("models/lithora_model.keras")

print("\nNew model saved:")
print(MODEL_PATH)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Lithora CNN Learning Curve")
plt.legend()
plt.grid(True)

plt.savefig("models/learning_curve.png", dpi=150)
plt.show()

print("Learning curve saved to models/learning_curve.png")
