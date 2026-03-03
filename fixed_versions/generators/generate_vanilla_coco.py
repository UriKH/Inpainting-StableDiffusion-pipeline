import os
import sys
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

# הוספת תיקיית האב לנתיב
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vanilla_pipeline import InpaintPipeline
from pipeline import InpaintPipelineInput

def load_coco_data(instances_json_path, captions_json_path):
    """
    טוען גם את ה-Bounding Boxes וגם את ה-Captions של התמונות.
    """
    # 1. טעינת Instances (מיקומי אובייקטים וקטגוריות)
    with open(instances_json_path, 'r') as f:
        instances_data = json.load(f)
        
    img_filename_to_id = {img['file_name']: img['id'] for img in instances_data['images']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in instances_data['categories']}
    
    img_id_to_ann = {}
    for ann in instances_data['annotations']:
        if ann['image_id'] not in img_id_to_ann:
            img_id_to_ann[ann['image_id']] = ann
            
    # 2. טעינת Captions (תיאורים גלובליים לתמונות)
    with open(captions_json_path, 'r') as f:
        captions_data = json.load(f)
        
    img_id_to_caption = {}
    for cap in captions_data['annotations']:
        if cap['image_id'] not in img_id_to_caption:
            # ניקח את התיאור הראשון וננקה נקודה בסוף משפט ורווחים מיותרים
            img_id_to_caption[cap['image_id']] = cap['caption'].strip().rstrip('.')
            
    return img_filename_to_id, img_id_to_ann, cat_id_to_name, img_id_to_caption

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    val_images_dir = os.path.join(base_dir, 'data', 'coco', 'validation')
    output_dir = os.path.join(base_dir, 'data', 'coco', 'validation_vanilla_results')
    
    # נתיבים לשני קובצי ה-JSON הנדרשים (ודא ששניהם קיימים בתיקייה זו)
    instances_file = os.path.join(base_dir, 'data', 'coco', 'annotations', 'instances_val2017.json') 
    captions_file = os.path.join(base_dir, 'data', 'coco', 'annotations', 'captions_val2017.json') 
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading COCO annotations and captions...")
    img_filename_to_id, img_id_to_ann, cat_id_to_name, img_id_to_caption = load_coco_data(instances_file, captions_file)
    
    print("Loading Vanilla Pipeline...")
    pipeline = InpaintPipeline() 
    
    image_files = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:150]

    for filename in tqdm(image_files, desc="Processing COCO validation images"):
        if filename not in img_filename_to_id:
            continue
            
        img_id = img_filename_to_id[filename]
        # נוודא שיש לנו גם אובייקט וגם תיאור לתמונה הזו
        if img_id not in img_id_to_ann or img_id not in img_id_to_caption:
            continue
            
        # 1. חילוץ המידע מה-JSON
        ann = img_id_to_ann[img_id]
        bbox = ann['bbox'] 
        category_name = cat_id_to_name[ann['category_id']]
        global_caption = img_id_to_caption[img_id]
        
        # 2. יצירת הפרומפט החדש בתבנית המבוקשת
        prompt = f"{category_name}, perfectly integrated into a scene of {global_caption}"
        
        # 3. טעינת התמונה המקורית
        img_path = os.path.join(val_images_dir, filename)
        init_image = Image.open(img_path).convert("RGB")
        
        # 4. יצירת מסיכה מלבנית
        mask_image = Image.new("L", init_image.size, 0)
        draw = ImageDraw.Draw(mask_image)
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], fill=255)
        
        # 5. הכנת האובייקט להרצה בפייפליין
        pipe_in = InpaintPipelineInput(prompt, init_image, mask_image)
        
        # 6. הרצת המודל
        result_img = pipeline.resize_pipe(pipe_in) 
            
        # 7. שמירת התוצאה
        out_path = os.path.join(output_dir, f"vanilla_{filename}")
        result_img.save(out_path)

if __name__ == "__main__":
    main()