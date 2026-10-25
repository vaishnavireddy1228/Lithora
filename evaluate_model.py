
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

df = pd.read_csv("dataset/image_manifest.csv")
test_df = df[df["group"] == "test"].copy()

label_map = {"Non-Stone": 0, "Stone": 1}
test_df["label_id"] = test_df["label"].map(label_map)

missing = [
    p for p in test_df["image_path"]
    if not Path(p).is_file()
]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} test images are missing. Example: {missing[0]}"
    )

def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=1)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

test_ds = tf.data.Dataset.from_tensor_slices((
    test_df["image_path"].to_numpy(),
    test_df["label_id"].to_numpy(dtype="int32")
))
test_ds = test_ds.map(load_image, num_parallel_calls=2)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)

model = tf.keras.models.load_model("models/lithora_model.keras")

# Model outputs a Stone score; 0.5 is the current threshold.
probabilities = model.predict(test_ds, verbose=1).ravel()
y_true = test_df["label_id"].to_numpy()
y_pred = (probabilities >= 0.5).astype(int)

print("\nConfusion matrix:")
print("Rows = actual, columns = predicted")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Non-Stone", "Stone"],
    zero_division=0
))

display = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=[0, 1],
    display_labels=["Non-Stone", "Stone"],
    values_format="d"
)
plt.title("Lithora Original CNN - Test Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.show()

print("\nSaved chart: models/confusion_matrix.png")
