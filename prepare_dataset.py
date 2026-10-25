from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

dataset_path=Path(r"C:\Users\vaish_6mpfxqm\Downloads\AXIALC~1\AXIALC~1\KINDEY~1\KINDEY~1\Original")

classes=["Stone","Non-Stone"]

# Collect each patient's ID and class.

patient_labels={}

for class_name in classes:
    class_folder=dataset_path/class_name

    for image_file in class_folder.glob("*.jpg"):
        patient_id=image_file.stem.split("_")[0]

        if patient_id in patient_labels:
            if patient_labels[patient_id] != class_name:
                raise ValueError(
                    f"Patient {patient_id} appears in both classes."
                )
        else:
            patient_labels[patient_id] = class_name

patients = list(patient_labels.keys())
labels = [patient_labels[p] for p in patients]

# First, reserve 15% for testing.
train_val_patients, test_patients, train_val_labels, test_labels = (
    train_test_split(
        patients,
        labels,
        test_size=0.15,
        random_state=42,
        stratify=labels
    )
)

# Split the remaining 85% into training and validation.
train_patients, val_patients, train_labels, val_labels = (
    train_test_split(
        train_val_patients,
        train_val_labels,
        test_size=0.15 / 0.85,
        random_state=42,
        stratify=train_val_labels
    )
)

print("Unique patients:", len(patients))
print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Testing patients:", len(test_patients))

print("\nPatient overlap checks:")
print("Train/validation:", bool(set(train_patients) & set(val_patients)))
print("Train/test:", bool(set(train_patients) & set(test_patients)))
print("Validation/test:", bool(set(val_patients) & set(test_patients)))

print("\nTraining class counts:", Counter(train_labels))
print("Validation class counts:", Counter(val_labels))
print("Testing class counts:", Counter(test_labels))