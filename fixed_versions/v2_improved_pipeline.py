from vanilla_pipeline import InpaintPipeline, InpaintPipelineInput
import torch
from PIL import Image
import cv2 as cv
import numpy as np


class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self):
        super().__init__()

    def image_preprocessing(self, real_image: Image.Image, mask_image: Image.Image):
        img_arr = np.asarray(real_image)
        mask_arr = np.asarray(mask_image)

        if len(mask_arr.shape) == 3:
            mask_arr = mask_arr[:, :, 0]

        smoothed_img_arr = cv.inpaint(img_arr, mask_arr, 3, cv.INPAINT_TELEA)
        return Image.fromarray(smoothed_img_arr)

    def encode_prompt(self, prompt, text_encoder, tokenizer):
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = tokenizer(
            [
                "ugly, tiling, poorly drawn, out of frame, deformed, blurry, bad anatomy, bad proportions, extra limbs,"
                " artifacts, miniature scene, entire picture, out of context, mismatched lighting"
            ],
            padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def postprocess(self, image: Image.Image, mask: Image.Image):
        return image
