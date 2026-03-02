import os
import json
from PIL import Image, ImageDraw
from ready_pipeline import PipeLine
from pipeline import InpaintPipelineInput

def run_10_coco_images():
    base_dir = 'data/coco'
    img_dir = os.path.join(base_dir, 'train2017_subset')
    out_dir = os.path.join(base_dir, 'ready_results')
    
    # יצירת תיקייה לתוצאות
    os.makedirs(out_dir, exist_ok=True)

    captions_file = os.path.join(base_dir, 'annotations/captions_train2017.json')
    instances_file = os.path.join(base_dir, 'annotations/instances_train2017.json')

    print("Loading annotations...")
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    with open(instances_file, 'r') as f:
        instances_data = json.load(f)

    # מיפוי מזהה תמונה לפרומפט
    captions_dict = {}
    for ann in captions_data['annotations']:
        img_id = ann['image_id']
        if img_id not in captions_dict:
            captions_dict[img_id] = ann['caption'] # ניקח את התיאור הראשון

    # מיפוי מזהה תמונה למיקומי אובייקטים (לצורך המסכה)
    instances_dict = {}
    for ann in instances_data['annotations']:
        img_id = ann['image_id']
        if img_id not in instances_dict:
            instances_dict[img_id] = []
        instances_dict[img_id].append(ann)

    print("Initializing Stable Diffusion Pipeline...")
    pipeline = PipeLine()

    # לוקחים את 10 התמונות הראשונות
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:10]

    for i, filename in enumerate(images):
        print(f"\nProcessing {i+1}/10: {filename}")
        img_path = os.path.join(img_dir, filename)
        image_id = int(filename.split('.')[0])

        prompt = captions_dict.get(image_id, "a photo of an object")
        print(f"Prompt: {prompt}")
        
        # טעינת התמונה ושינוי גודל ל-512x512 (הגודל ש-Stable Diffusion מצפה לקבל)
        init_image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = init_image.size
        init_image = init_image.resize((512, 512))

        # יצירת תמונת מסכה שחורה
        mask_image = Image.new("RGB", (orig_width, orig_height), (0, 0, 0))
        draw = ImageDraw.Draw(mask_image)
        
        # ציור מלבן לבן (מסכה) על האובייקט הראשון שזוהה בתמונה
        anns = instances_dict.get(image_id, [])
        if anns:
            bbox = anns[0]['bbox'] # [x, y, width, height]
            x, y, w, h = bbox
            draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255))
        else:
            # אם אין אובייקטים מתויגים, ניצור מסכה ריבועית באמצע התמונה כברירת מחדל
            draw.rectangle((orig_width//4, orig_height//4, orig_width*3//4, orig_height*3//4), fill=(255, 255, 255))

        # התאמת גודל המסכה לגודל התמונה
        mask_image = mask_image.resize((512, 512))

        # הרצת הפייפליין שלך
        pipe_in = InpaintPipelineInput(prompt, init_image, mask_image)
        result = pipeline.pipe(pipe_in)

        # שמירת התוצאה
        out_path = os.path.join(out_dir, f"result_{filename}")
        result.save(out_path)

    print(f"\nDone! Check the '{out_dir}' folder for the generated images.")

if __name__ == "__main__":
    run_10_coco_images()