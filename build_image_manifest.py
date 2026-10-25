from pathlib import Path
import csv

dataset_path = Path(
    r"C:\Users\vaish_6mpfxqm\Downloads\AXIALC~1\AXIALC~1\KINDEY~1\KINDEY~1\Original"
)

# Get the same patient split each time.
# Import the split logic from our existing script.
from prepare_dataset import (
    patient_labels,
    train_patients,
    val_patients,
    test_patients
)

patient_groups = {}

for patient_id in train_patients:
    patient_groups[patient_id] = "train"

for patient_id in val_patients:
    patient_groups[patient_id] = "validation"

for patient_id in test_patients:
    patient_groups[patient_id] = "test"

rows = []

for class_name in ["Stone", "Non-Stone"]:
    class_folder = dataset_path / class_name

    for image_file in class_folder.glob("*.jpg"):
        patient_id = image_file.stem.split("_")[0]

        rows.append({
            "image_path": str(image_file),
            "patient_id": patient_id,
            "label": class_name,
            "group": patient_groups[patient_id]
        })

output_path = Path("dataset") / "image_manifest.csv"

with output_path.open("w", newline="", encoding="utf-8") as csv_file:
    fieldnames = ["image_path", "patient_id", "label", "group"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)

print("Manifest saved to:", output_path)
print("Total image records:", len(rows))
print("CSV columns:", fieldnames)
   