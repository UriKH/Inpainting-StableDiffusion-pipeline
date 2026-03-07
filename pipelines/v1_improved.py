from pipelines.vanilla_pipeline import InpaintPipelineVanilla
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np


class ImprovedInpaintPipelineV1(InpaintPipelineVanilla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
    def image_preprocessing(self, real_image, mask_image):
        real_arr = np.array(real_image)
        mask_arr = np.array(mask_image)
        mask_bool = mask_arr == 255
        
        filled_arr = real_arr.copy()
        mean_bg_color = np.mean(real_arr[~mask_bool], axis=0)
        filled_arr[mask_bool] = mean_bg_color
        
        iterations = 15
        blur_kernel = (15, 15)
        
        for _ in range(iterations):
            blurred = cv.GaussianBlur(filled_arr, blur_kernel, 0)
            filled_arr[mask_bool] = blurred[mask_bool]
        return Image.fromarray(filled_arr)