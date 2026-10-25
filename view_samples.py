
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cv2

df = pd.read_csv("dataset/image_manifest.csv")

samples = []

for label in ["Non-Stone", "Stone"]:
    subset = df[
        (df["label"] == label) &
        (df["group"] == "train")
    ].sample(n=4, random_state=42)

    for _, row in subset.iterrows():
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)

        if image is not None:
            samples.append((label, image, Path(row["image_path"]).name))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for ax, (label, image, filename) in zip(axes.flat, samples):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"{label}\n{filename}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()
