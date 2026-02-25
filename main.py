import torch
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel, 
    AutoencoderKL, 
    DDPMScheduler  # Make sure to import DDPMScheduler instead of PNDMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw

def load_sd2_components(model_path, device="cuda"):
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae",
                                        torch_dtype=torch.float32).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet",
                                                torch_dtype=torch.float32).to(device)
    text_encoder = CLIPTextModel.from_pretrained(model_path,subfolder="text_encoder",
                                                 torch_dtype=torch.float32).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(model_path,subfolder="scheduler",
                                              torch_dtype=torch.float32)
    print("All components loaded successfully!")
    return vae, unet, text_encoder, tokenizer, scheduler

# The exact model requested in the assignment instructions 
model_id = "Manojb/stable-diffusion-2-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load components using your provided function
print("Loading components...")
vae, unet, text_encoder, tokenizer, scheduler = load_sd2_components(model_id, device=device)

# 2. Combine them into the Inpainting Pipeline
print("Assembling pipeline...")
pipe = StableDiffusionInpaintPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,     
    feature_extractor=None,
)

# Since your function already moved the individual components to the device, 
# we just make sure the pipeline wrapper itself knows which device it's on.
pipe = pipe.to(device)

# 3. Create dummy data for a quick sanity check
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

# Save the result
result.save("vanilla_test_output.png")
print("Done! Check vanilla_test_output.png")
