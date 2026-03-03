import os
import random
import shutil


def split_folder_data(source_folder, eval_folder, val_folder, split_ratio=0.9):
    # 1. Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return

    # 2. Get all files from the source directory (ignoring subdirectories)
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if not all_files:
        print("The source folder is empty.")
        return

    # 3. Shuffle the files randomly
    random.seed(42) # Optional: remove or change this for different random splits each time
    random.shuffle(all_files)

    # 4. Calculate the split index
    split_index = int(len(all_files) * split_ratio)

    # 5. Split into disjoint sets
    eval_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # 6. Create the destination folders if they don't exist
    os.makedirs(eval_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 7. Copy files to the Evaluation folder
    print(f"Copying {len(eval_files)} files to '{eval_folder}'...")
    for file_name in eval_files:
        source_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(eval_folder, file_name)
        shutil.copy2(source_path, dest_path) # copy2 preserves file metadata

    # 8. Copy files to the Validation folder
    print(f"Copying {len(val_files)} files to '{val_folder}'...")
    for file_name in val_files:
        source_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(val_folder, file_name)
        shutil.copy2(source_path, dest_path)

    print("Done! Files successfully split into disjoint sets.")


# --- Example Usage ---
if __name__ == "__main__":
    # Update these paths to match your local setup
    SOURCE_DIR = "../data/coco/val2017_subset"
    EVAL_DIR = "../data/coco/evaluation"
    VAL_DIR = "../data/coco/validation"
    
    split_folder_data(SOURCE_DIR, EVAL_DIR, VAL_DIR, split_ratio=0.85)
