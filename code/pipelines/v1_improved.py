from pipelines.vanilla_pipeline import InpaintPipelineVanilla
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np


class ImprovedInpaintPipelineV1(InpaintPipelineVanilla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
    def image_preprocessing(self, real_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        """
        Fills the blacked out regions of the image with the mean color of the non-black pixels and iteratively blurs the image.
        :param real_image: The image to be inpainted.
        :param mask_image: The mask image to guide the inpainting process.
        :return: The filled image.
        """
        img_arr = np.array(real_image)
        mask_bool = np.array(mask_image) == 255
        filled = img_arr.copy()
        filled[mask_bool] = np.mean(img_arr[~mask_bool], axis=0)
        iterations = 15
        blur_kernel = (15, 15)
        
        for _ in range(iterations):
            blurred = cv.GaussianBlur(filled, blur_kernel, 0)
            filled[mask_bool] = blurred[mask_bool]
        return Image.fromarray(filled)
