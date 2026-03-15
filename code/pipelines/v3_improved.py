from code.pipelines.vanilla_pipeline import InpaintPipelineVanilla
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt


class ImprovedInpaintPipelineV3(InpaintPipelineVanilla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
    def image_preprocessing(self, real_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        """
        Fills the blacked-out regions of the image with the nearest non-masked pixel and blurs the image.
        :param real_image: The image to be inpainted.
        :param mask_image: The mask image to guide the inpainting process.
        :return: The filled image.
        """
        img_arr = np.array(real_image)
        mask_bool = np.array(mask_image) == 255
        _, indices = distance_transform_edt(mask_bool, return_indices=True)
        
        filled = img_arr.copy()
        filled[mask_bool] = img_arr[tuple(indices[:, mask_bool])]
        blurred = cv.GaussianBlur(filled, (21, 21), 0)
        filled[mask_bool] = blurred[mask_bool]
        return Image.fromarray(filled)
