from pathlib import Path

dataset_path=Path(r"C:\Users\vaish_6mpfxqm\Downloads\AXIALC~1\AXIALC~1\KINDEY~1\KINDEY~1\Original")

stone_folder=dataset_path / "Stone"
non_stone_folder=dataset_path / "Non-Stone"

print("Stone folder exists:", stone_folder.is_dir())
print("Non-Stone folder exists:", non_stone_folder.is_dir())

if stone_folder.is_dir() and non_stone_folder.is_dir():
    image_extensions={".jpg",".jpeg",".png"}
    stone_images=[f for f in stone_folder.iterdir()
                  if f.is_file() and f.suffix.lower() in image_extensions]

    non_stone_images=[f for f in non_stone_folder.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]
    print("Stone files:",len(stone_images))
    print("Non-Stone files:",len(non_stone_images))
    print("Total files:",len(stone_images)+len(non_stone_images))
else:
    print("Please check the dataset path and folder names.")
print("\nSample Stone filenames:")
for image_file in stone_images[:5]:
    print(image_file.name)

print("\nSample Non-Stone filenames:")
for image_file in non_stone_images[:5]:
    print(image_file.name)

def get_patient_id(image_file):
    return image_file.stem.split("_")[0]
stone_patients={get_patient_id(f) for f in stone_images}
non_stone_patients={get_patient_id(f) for f in non_stone_images}
overlap=stone_patients&non_stone_patients
print("\nUnique Stone patient IDs:", len(stone_patients))
print("Unique Non-Stone patient IDs:", len(non_stone_patients))
print("Patient IDs present in both classes:", len(overlap))
if overlap:
    print("Example overlapping IDs:", sorted(overlap)[:10])