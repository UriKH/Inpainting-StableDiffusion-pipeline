import os
import json
from PIL import Image, ImageDraw

# הגדרת נתיבים
base_dir = "data/coco"
img_dir = os.path.join(base_dir, "val2017_subset")
out_dir = os.path.join(base_dir, "sample_masks_10")
instances_file = os.path.join(base_dir, "annotations/instances_val2017.json")

# יצירת תיקיית הפלט אם היא לא קיימת
os.makedirs(out_dir, exist_ok=True)

print("Loading instances JSON...")
with open(instances_file, 'r') as f:
    instances_data = json.load(f)

# יצירת מילון לחיפוש מהיר של הקואורדינטות לפי ה-ID של התמונה
instances_dict = {}
for ann in instances_data['annotations']:
    img_id = ann['image_id']
    if img_id not in instances_dict:
        instances_dict[img_id] = []
    instances_dict[img_id].append(ann)

# לקיחת 10 התמונות הראשונות מהתיקייה
images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:10]

for filename in images:
    image_id = int(filename.split('.')[0])
    img_path = os.path.join(img_dir, filename)
    
    # פתיחת התמונה המקורית רק כדי לקבל את המידות שלה (רוחב וגובה)
    init_image = Image.open(img_path)
    orig_width, orig_height = init_image.size
    
    # יצירת קנבס שחור
    mask_image = Image.new("RGB", (orig_width, orig_height), (0, 0, 0))
    draw = ImageDraw.Draw(mask_image)
    
    # ציור המלבן הלבן על סמך הנתונים מ-COCO
    anns = instances_dict.get(image_id, [])
    if anns:
        bbox = anns[0]['bbox']
        x, y, w, h = bbox
        draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255))
    else:
        # אם במקרה אין תיוג, נצייר מלבן גנרי באמצע
        draw.rectangle((orig_width//4, orig_height//4, orig_width*3//4, orig_height*3//4), fill=(255, 255, 255))
        
    # שמירת המסכה
    mask_filename = filename.replace('.jpg', '.mask.png')
    mask_image.save(os.path.join(out_dir, mask_filename))
    print(f"Saved: {mask_filename}")

print(f"\nDone! Saved 10 masks to '{out_dir}'.")