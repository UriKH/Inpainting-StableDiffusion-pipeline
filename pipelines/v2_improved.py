from pipelines.vanilla_pipeline import InpaintPipelineVanilla
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np


class ImprovedInpaintPipelineV2(InpaintPipelineVanilla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
    def image_preprocessing(self, real_image, mask_image):
        real_arr = np.array(real_image)
        mask_arr = np.array(mask_image)
        
        mask_bool = mask_arr == 255
        _, indices = distance_transform_edt(mask_bool, return_indices=True)
        
        filled_arr = real_arr.copy()
        filled_arr[mask_bool] = real_arr[tuple(indices[:, mask_bool])]
        return Image.fromarray(filled_arr)