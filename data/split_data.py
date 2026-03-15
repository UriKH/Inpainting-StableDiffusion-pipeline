import os
import random
import shutil


def split_folder_data(source_folder, eval_folder, val_folder, split_ratio=0.9):
    if not os.path.exists(source_folder):
        raise Exception(f"Error: Source folder '{source_folder}' does not exist.")

    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if not all_files:
        raise Exception("The source folder is empty.")

    random.seed(42)
    random.shuffle(all_files)

    split_index = int(len(all_files) * split_ratio)
    eval_files = all_files[:split_index]
    val_files = all_files[split_index:]

    os.makedirs(eval_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    print(f"Copying {len(eval_files)} files to '{eval_folder}'...")
    for file_name in eval_files:
        source_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(eval_folder, file_name)
        shutil.copy2(source_path, dest_path)

    print(f"Copying {len(val_files)} files to '{val_folder}'...")
    for file_name in val_files:
        source_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(val_folder, file_name)
        shutil.copy2(source_path, dest_path)
    print("Done! Files successfully split into disjoint sets.")


if __name__ == "__main__":
    SOURCE_DIR = "./data/coco/val2017_subset"
    EVAL_DIR = "./data/coco/evaluation"
    VAL_DIR = "./data/coco/validation"
    split_folder_data(SOURCE_DIR, EVAL_DIR, VAL_DIR, split_ratio=5/6.0)
