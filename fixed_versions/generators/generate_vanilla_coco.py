import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from vanilla_pipeline import InpaintPipeline
from pipeline import InpaintPipelineInput

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    img_filename_to_id = {img['file_name']: img['id'] for img in data['images']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    img_id_to_ann = {}
    for ann in data['annotations']:
        if ann['image_id'] not in img_id_to_ann:
            img_id_to_ann[ann['image_id']] = ann
            
    return img_filename_to_id, img_id_to_ann, cat_id_to_name


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    val_images_dir = os.path.join(base_dir, 'data', 'coco', 'validation')
    output_dir = os.path.join(base_dir, 'data', 'coco', 'validation_vanilla_results')
    annotations_file = os.path.join(base_dir, 'data', 'coco', 'annotations', 'instances_val2017.json') 
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading COCO annotations...")
    img_filename_to_id, img_id_to_ann, cat_id_to_name = load_coco_annotations(annotations_file)
    
    print("Loading Vanilla Pipeline...")
    pipeline = InpaintPipeline() 
    
    # רשימת התמונות (רק קבצי תמונה)
    image_files = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # הגבלה ל-150 תמונות בדיוק, כמו שביקשת
    image_files = image_files[:150]

    for filename in tqdm(image_files, desc="Processing COCO validation images"):
        if filename not in img_filename_to_id:
            continue
            
        img_id = img_filename_to_id[filename]
        if img_id not in img_id_to_ann:
            continue
            
        # 1. חילוץ המידע מה-JSON
        ann = img_id_to_ann[img_id]
        bbox = ann['bbox'] # מגיע בפורמט של [x, y, width, height]
        category_name = cat_id_to_name[ann['category_id']]
        
        # 2. יצירת פרומפט בסיסי (אפשר לשנות בהתאם לצורך)
        prompt = f"a photo of a {category_name}"
        
        # 3. טעינת התמונה המקורית
        img_path = os.path.join(val_images_dir, filename)
        init_image = Image.open(img_path).convert("RGB")
        
        # 4. יצירת מסיכה מלבנית לפי ה-Bounding Box
        # המסיכה היא תמונת שחור-לבן (L). הפיקסלים שמיועדים ל-Inpainting יהיו לבנים (255)
        mask_image = Image.new("L", init_image.size, 0)
        draw = ImageDraw.Draw(mask_image)
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], fill=255)
        
        # 5. הכנת האובייקט להרצה בפייפליין
        pipe_in = InpaintPipelineInput(prompt, init_image, mask_image)
        
        # 6. הרצת המודל. 
        # נשתמש ב-resize_pipe מתוך ההורשה מ-SD2InpaintingPipeLineScheme כדי לטפל בגדלים
        result_img = pipeline.resize_pipe(pipe_in) 
            
        # 7. שמירת התוצאה
        out_path = os.path.join(output_dir, f"vanilla_{filename}")
        result_img.save(out_path)