import dataclasses
import os

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from pipeline import SD2InpaintingPipeLineScheme, InpaintPipelineInput


MODEL_ID = "Manojb/stable-diffusion-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PipeLine(SD2InpaintingPipeLineScheme):
    def __init__(self, model_id=MODEL_ID, device=DEVICE):
        super().__init__(model_id, device)
        self.p = StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)

    def pipe(self, pipe_in: InpaintPipelineInput):
        image = self.p(prompt=pipe_in.prompt, image=pipe_in.init_image, mask_image=pipe_in.mask_image).images[0]
        return image


if __name__ == "__main__":
    p = PipeLine(MODEL_ID, DEVICE)

    init_image = Image.new("RGB", (512, 512), (200, 200, 200))  # A plain gray square
    mask_image = Image.new("RGB", (512, 512), (0, 0, 0))  # Black mask
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle((200, 200, 312, 312), fill=(255, 255, 255))  # White square in the middle
    prompt = "a bright red apple"

    result = p.pipe(InpaintPipelineInput(prompt, init_image, mask_image))
    result.save("vanilla_test_output.png")
    print("Success!")
