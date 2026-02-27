import dataclasses
import os

import torch
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel, 
    AutoencoderKL, 
    DDPMScheduler
)

from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw
from dataclasses import dataclass

MODEL_ID = "Manojb/stable-diffusion-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TripletInpaintPipeline:
    prompt: str
    image: Image.Image
    mask_image: Image.Image


class PipeLine:
    def __init__(self, model_id, device):
        self.vae, self.unet, self.text_encoder, self.tokenizer, self.scheduler = self.load_sd2_components(model_id, device=device)
        self.pipe = StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

    @staticmethod
    def load_sd2_components(model_path, device="cuda"):
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).to(device)
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float32).to(device)
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                                     torch_dtype=torch.float32).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch.float32)
        print(f'{"=" * 10} Components Loaded {"=" * 10}')
        return vae, unet, text_encoder, tokenizer, scheduler

    def apply(self, pipe_in: TripletInpaintPipeline):
        return self.pipe(**dataclasses.asdict(pipe_in)).images[0]

    def apply_multiple(self, dir_in: str, dir_out: str):
        file_names = {file.split('.')[0] for file in os.listdir(dir_in)}
        for name in file_names:
            if len(name) == 0:
                continue
            prompt = None
            with open(os.path.join(dir_in, name, '.txt'), "r") as f:
                prompt = f.read()

            mask = Image.open(os.path.join(dir_in, name, '.mask.png')).convert('RGB')
            image = Image.open(os.path.join(dir_in, name, '.png')).convert('RGB')
            pipe_in = TripletInpaintPipeline(prompt=prompt, image=image, mask_image=mask)
            result = self.apply(pipe_in)
            result.save(os.path.join(dir_out, name, '.png'))


if __name__ == "__main__":
    p = PipeLine(MODEL_ID, DEVICE)

    init_image = Image.new("RGB", (512, 512), (200, 200, 200))  # A plain gray square
    mask_image = Image.new("RGB", (512, 512), (0, 0, 0))  # Black mask
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle((200, 200, 312, 312), fill=(255, 255, 255))  # White square in the middle
    prompt = "a bright red apple"

    result = p.apply(TripletInpaintPipeline(prompt=prompt, image=init_image, mask_image=mask_image))
    result.save("vanilla_test_output.png")
    print("Success!")