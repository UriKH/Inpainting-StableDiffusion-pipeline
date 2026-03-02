import os
import json
import urllib.request
import zipfile

def download_coco_subset(num_images=1000):
    base_dir = 'data/coco'
    ann_zip_path = os.path.join(base_dir, 'annotations_trainval2017.zip')
    # שימוש בקובצי ה-val הקטנים משמעותית
    ann_file = os.path.join(base_dir, 'annotations/instances_val2017.json')
    out_dir = os.path.join(base_dir, 'val2017_subset')
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ann_file):
        if not os.path.exists(ann_zip_path):
            print("Downloading annotations zip (approx 241MB)...")
            urllib.request.urlretrieve('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', ann_zip_path)
        
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
            
        os.remove(ann_zip_path)

    print("Loading annotations (val set is small and fast!)...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    images_to_download = coco_data['images'][:num_images]

    print(f"Downloading {num_images} images...")
    for i, img in enumerate(images_to_download):
        url = img['coco_url']
        filename = img['file_name']
        filepath = os.path.join(out_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Downloading {i+1}/{num_images}: {filename}")
            urllib.request.urlretrieve(url, filepath)
            
    print("All done!")

if __name__ == "__main__":
    download_coco_subset(20)
