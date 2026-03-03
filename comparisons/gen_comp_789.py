import os
import importlib.util
from pathlib import Path
from PIL import Image
import numpy as np
from old.pipeline import InpaintPipelineInput

# Import the official COCO API
from pycocotools.coco import COCO
from tqdm import tqdm


# 2. Helper to dynamically load classes
def load_pipeline_class(filepath):
    print(f'LOADING: {filepath}')
    module_name = Path(filepath).stem
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ImprovedInpaintPipeline


# 3. The Main Evaluation Loop
def run_coco_benchmark(pipelines_dir, val_images_dir, annotations_dir, output_dir):
    print(f'>>> Running benchmarks... <<<')
    pipelines_dir = Path(pipelines_dir)
    val_images_dir = Path(val_images_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)

    # --- INITIALIZE COCO API ---
    print("Loading COCO Annotations into memory (this takes a few seconds)...")
    instances_json = annotations_dir / "instances_val2017.json"
    captions_json = annotations_dir / "captions_val2017.json"
    
    coco_instances = COCO(str(instances_json))
    coco_caps = COCO(str(captions_json))
    print("COCO API loaded successfully!\n")

    # Find all python files in the pipelines directory
    pipeline_files = list(pipelines_dir.glob("*.py"))
    if not pipeline_files:
        print(f"No pipeline files found in {pipelines_dir}")
        return

    # Get valid images from your specific validation split folder
    image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for pipe_file in pipeline_files:
        version_name = pipe_file.stem
        print(f"--- Running Pipeline Version: {version_name} ---")
        
        PipelineClass = load_pipeline_class(pipe_file)
        pipeline_instance = PipelineClass()
        
        version_out_dir = output_dir / version_name
        version_out_dir.mkdir(parents=True, exist_ok=True)

        for img_name in tqdm(image_files, desc=f'running on pipe: {version_name}'):
            # 1. Parse COCO Image ID from the filename (e.g., '000000397133.jpg' -> 397133)
            try:
                img_id = int(os.path.splitext(img_name)[0])
            except ValueError:
                print(f"  Warning: Could not parse COCO ID from filename {img_name}. Skipping.")
                continue

            # 2. Extract Prompt (Using the first available caption)
            ann_ids_caps = coco_caps.getAnnIds(imgIds=img_id)
            if not ann_ids_caps:
                print(f"  Warning: No captions found for {img_name}. Skipping.")
                continue
            captions = coco_caps.loadAnns(ann_ids_caps)
            prompt = captions[0]['caption'] 

            # 3. Extract Mask (Picking the largest object annotation in the image)
            ann_ids_inst = coco_instances.getAnnIds(imgIds=img_id)
            if not ann_ids_inst:
                print(f"  Warning: No instance annotations found for {img_name}. Skipping.")
                continue
            
            instances = coco_instances.loadAnns(ann_ids_inst)
            # Find the largest annotation by area to give the inpainting pipeline a good target
            largest_instance = max(instances, key=lambda ann: ann['area'])
            
            # Use COCO API to generate a binary numpy mask (0s and 1s)
            mask_np = coco_instances.annToMask(largest_instance)
            # Convert to PIL Image (0 and 255)
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

            # 4. Load Image
            init_img_path = val_images_dir / img_name
            init_img = Image.open(init_img_path).convert("RGB")

            # 5. Package Input and Run
            pipe_in = InpaintPipelineInput(init_image=init_img, mask_image=mask_img, prompt=prompt)

            try:
                result_img = pipeline_instance.pipe(pipe_in)
                save_path = version_out_dir / img_name
                result_img.save(save_path)
                print(f"  Processed {img_name} | Prompt: '{prompt[:30]}...'")
            except Exception as e:
                print(f"  Error processing {img_name} with {version_name}: {e}")

    print("\nBenchmark complete! Results saved to:", output_dir)

# --- Configuration ---
if __name__ == "__main__":
    PIPELINES_FOLDER = "./comp_789/pipeline_versions_to_test" 
    VAL_IMAGES = "../data/coco/validation"           # Your 10% subset
    ANNOTATIONS_DIR = "../data/coco/annotations"            # Folder containing the COCO JSONs
    OUTPUT_FOLDER = "./results"

    run_coco_benchmark(PIPELINES_FOLDER, VAL_IMAGES, ANNOTATIONS_DIR, OUTPUT_FOLDER)
