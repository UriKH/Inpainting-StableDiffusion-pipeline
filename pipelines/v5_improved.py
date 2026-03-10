from pipelines.v3_improved import ImprovedInpaintPipelineV3
from pipelines.pipeline import InpaintPipelineInput
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np


class ImprovedInpaintPipelineV5(ImprovedInpaintPipelineV3):
    def __init__(self, pp_dilate_kernel_size=15, pp_feather_kernel_size=21, use_negative_prompt=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pp_dilate_kernel_size = pp_dilate_kernel_size
        self.pp_feather_kernel_size = pp_feather_kernel_size
        self.use_negative_prompt = use_negative_prompt

    def encode_prompt(self, prompt, text_encoder, tokenizer):
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = tokenizer(
            [
                "ugly, tiling, poorly drawn, out of frame, deformed, blurry, bad anatomy, bad proportions, extra limbs, artifacts, miniature scene, entire picture, out of context, mismatched lighting" if self.use_negative_prompt else ""
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
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.pp_dilate_kernel_size, self.pp_dilate_kernel_size))
        mask_dilated = cv.dilate(mask_np, kernel, iterations=1)
        if self.pp_feather_kernel_size != 1:
            feathered_mask = cv.GaussianBlur(mask_dilated, (self.pp_feather_kernel_size, self.pp_feather_kernel_size), 0)
        else:
            feathered_mask = mask_dilated
        return Image.fromarray(feathered_mask)

    def preprocess(self, pipe_in: InpaintPipelineInput):
        org_mask = pipe_in.mask_image
        pipe_in.mask_image = self.mask_preprocessing(pipe_in.mask_image)
        pipe_in.init_image = self.image_preprocessing(pipe_in.init_image, org_mask)
        return pipe_in
