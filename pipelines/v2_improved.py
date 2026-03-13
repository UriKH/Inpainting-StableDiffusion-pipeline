from pipelines.vanilla_pipeline import InpaintPipelineVanilla
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt


class ImprovedInpaintPipelineV2(InpaintPipelineVanilla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
    def image_preprocessing(self, real_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        """
        Fills the blacked-out regions of the image with the nearest non-masked pixel.
        :param real_image: The image to be inpainted.
        :param mask_image: The mask image to guide the inpainting process.
        :return: The filled image.
        """
        real_arr = np.array(real_image)
        mask_arr = np.array(mask_image)
        mask_bool = mask_arr == 255

        _, indices = distance_transform_edt(mask_bool, return_indices=True)
        filled_arr = real_arr.copy()
        filled_arr[mask_bool] = real_arr[tuple(indices[:, mask_bool])]
        return Image.fromarray(filled_arr)
