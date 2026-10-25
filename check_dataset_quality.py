
import pandas as pd
import cv2
import numpy as np

df = pd.read_csv("dataset/image_manifest.csv")

def image_features(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(image, (128, 128))

    return {
        "mean_brightness": float(np.mean(image)),
        "contrast": float(np.std(image)),
        "dark_pixel_fraction": float(np.mean(image < 30)),
        "bright_pixel_fraction": float(np.mean(image > 220)),
    }

records = []

for _, row in df.iterrows():
    features = image_features(row["image_path"])

    if features is not None:
        records.append({
            "group": row["group"],
            "label": row["label"],
            **features
        })

features_df = pd.DataFrame(records)

print("\nImage counts by split and class:")
print(
    df.groupby(["group", "label"]).size().to_string()
)

print("\nImage characteristics by split and class:")
print(
    features_df.groupby(["group", "label"])[
        [
            "mean_brightness",
            "contrast",
            "dark_pixel_fraction",
            "bright_pixel_fraction"
        ]
    ].agg(["mean", "std"]).round(3).to_string()
)