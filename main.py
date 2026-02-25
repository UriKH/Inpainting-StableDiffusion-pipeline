import torch
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel, 
    AutoencoderKL, 
    PNDMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw

# The exact model requested in the assignment instructions 
model_id = "stabilityai/stable-diffusion-2-base"

# 1. Load each component individually
print("Loading components...")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

# 2. Combine them into the Inpainting Pipeline
print("Assembling pipeline...")
pipe = StableDiffusionInpaintPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,     # Optional: disabled for raw research testing
    feature_extractor=None,
)

# Move to GPU for faster processing
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# 3. Create dummy data for a quick sanity check
# Note: For your actual report, you must use your own images and masks! [cite: 68]
print("Generating dummy image and mask...")
init_image = Image.new("RGB", (512, 512), (200, 200, 200)) # A plain gray square
mask_image = Image.new("RGB", (512, 512), (0, 0, 0))       # Black mask
draw = ImageDraw.Draw(mask_image)
draw.rectangle((200, 200, 312, 312), fill=(255, 255, 255)) # White square in the middle

# 4. Run the model
prompt = "a bright red apple"
print(f"Running inference for prompt: '{prompt}'")

# The pipeline requires the image, the mask, and the prompt
result = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

# Save or display the result
result.save("vanilla_test_output.png")
print("Done! Check vanilla_test_output.png")
