from pipelines.vanilla_pipeline import InpaintPipelineVanilla
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np


class ImprovedInpaintPipelineV1(InpaintPipelineVanilla):
    def __init__(self, model_id=InpaintPipelineVanilla.MODEL_ID, device=InpaintPipelineVanilla.DEVICE, dilate_kernel_size=15, feather_radius=10):
        super().__init__(model_id, device)
        self.dilate_kernel_size = dilate_kernel_size
        self.feather_radius = feather_radius

    def encode_prompt(self, prompt, text_encoder, tokenizer):
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = tokenizer(
            [
                "ugly, tiling, poorly drawn, out of frame, deformed, blurry, bad anatomy, bad proportions, extra limbs, artifacts, miniature scene, entire picture, out of context, mismatched lighting"
            ],
            padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    def mask_preprocessing(self, mask_image: Image.Image):
        """
        Enhances the mask using Dilation and Feathering to prevent 'cut' edges.
        """
        mask_np = np.array(mask_image.convert("L"))
        kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
        mask_dilated = cv.dilate(mask_np, kernel, iterations=1)
        
        mask_pil = Image.fromarray(mask_dilated).filter(ImageFilter.GaussianBlur(radius=self.feather_radius))
        return mask_pil
    