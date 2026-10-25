
from pathlib import Path
from collections import defaultdict
import pandas as pd
import cv2

df = pd.read_csv("dataset/image_manifest.csv")

dimensions = defaultdict(list)

for _, row in df.iterrows():
    image = cv2.imread(row["image_path"])

    if image is not None:
        height, width = image.shape[:2]
        dimensions[row["label"]].append((width, height))

for label, sizes in dimensions.items():
    unique_sizes = set(sizes)

    print(f"\n{label}")
    print("Total readable images:", len(sizes))
    print("Different image dimensions:", len(unique_sizes))
    print("Most common dimensions:")

    counts = pd.Series(sizes).value_counts().head(5)

    for size, count in counts.items():
        print(f"  {size}: {count} images")

