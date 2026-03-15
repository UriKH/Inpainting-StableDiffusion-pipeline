from code.pipelines.v3_improved import ImprovedInpaintPipelineV3
from code.pipelines.pipeline import InpaintPipelineInput
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np


class ImprovedInpaintPipelineV5(ImprovedInpaintPipelineV3):
    def __init__(self, pp_dilate_kernel_size=3, pp_feather_radius=5, use_negative_prompt=True,
                 ignore_improvement_v5 = False, *args, **kwargs):
        """
        :param pp_dilate_kernel_size: The size of the kernel for mask dilation preprocessing.
        :param pp_feather_radius: The radius of the Gaussian blur for mask feathering preprocessing.
        :param use_negative_prompt: Whether to use a negative prompt.
        :param ignore_improvement_v5: Whether to ignore the improvements in V5.
        """
        super().__init__(*args, **kwargs)
        self.pp_dilate_kernel_size = pp_dilate_kernel_size
        self.pp_feather_radius = pp_feather_radius
        self.use_negative_prompt = use_negative_prompt
        self.ignore_improvement_v5 = ignore_improvement_v5

    def encode_prompt(self, prompt: str, encoder, tokenizer):
        """
        Encodes the prompt into text embeddings.
        :param prompt: The prompt to encode.
        :param encoder: The text encoder model.
        :param tokenizer: The tokenizer model.
        :return: The encoded prompt [uncond, text].
        """
        _, text_embeddings = super().encode_prompt(prompt, encoder, tokenizer).chunk(2)
        uncond_input = tokenizer(
            [
                "ugly, tiling, poorly drawn, out of frame, deformed, blurry, bad anatomy, bad proportions, extra limbs,"
                " artifacts, miniature scene, entire picture, out of context, mismatched lighting"
                if self.use_negative_prompt and not self.ignore_improvement_v5 else ""
            ],
            padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = encoder(uncond_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings])
    
    def mask_preprocessing(self, mask_image: Image.Image) -> Image.Image:
        """
        Enhances the mask using Dilation and Feathering to prevent 'cut' edges.
        :param mask_image: The mask image to enhance.
        :return: The enhanced mask image.
        """
        if self.ignore_improvement_v5:
            return super().mask_preprocessing(mask_image)

        mask_np = np.array(mask_image.convert("L"))
        kernel = np.ones((self.pp_dilate_kernel_size, self.pp_dilate_kernel_size), np.uint8)
        mask_dilated = cv.dilate(mask_np, kernel, iterations=1)
        if self.pp_feather_radius != 1:
            mask_pil = Image.fromarray(mask_dilated).filter(ImageFilter.GaussianBlur(radius=self.pp_feather_radius))
        else:
            mask_pil = Image.fromarray(mask_dilated)
        return mask_pil

    def preprocess(self, pipe_in: InpaintPipelineInput) -> InpaintPipelineInput:
        """
        Preprocesses the input data by applying mask enhancement and image enhancement.
        :param pipe_in: The input data.
        :return: The preprocessed input data.
        """
        if self.ignore_improvement_v5:
            return super().preprocess(pipe_in)

        org_mask = pipe_in.mask_image
        pipe_in.mask_image = self.mask_preprocessing(pipe_in.mask_image)
        pipe_in.init_image = self.image_preprocessing(pipe_in.init_image, org_mask)
        return pipe_in

