from code.pipelines.vanilla_pipeline import InpaintPipelineVanilla
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt


class ImprovedInpaintPipelineV4(InpaintPipelineVanilla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def image_preprocessing(self, real_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        """
        Fills the blacked-out regions of the image using the average of the surrounding pixels (computed iteratively).
        :param real_image: The image to be inpainted.
        :param mask_image: The mask image to guide the inpainting process.
        :return: The filled image.
        (This idea was inspired by a conversation with AI)
        """
        mask_bool = np.array(mask_image) == 255
        img_arr = np.array(real_image).astype(np.float32)
        filled = img_arr.copy()
        filled[mask_bool] = np.mean(img_arr[~mask_bool], axis=0)
        kernel = np.array(
            [[0.0,  0.25, 0.0],
            [0.25, 0.0,  0.25],
            [0.0,  0.25, 0.0]],
            dtype=np.float32
        )
        
        max_iters = 1000
        tolerance = 0.05
        for i in range(max_iters):
            smoothed = cv.filter2D(filled, -1, kernel, borderType=cv.BORDER_REPLICATE)
            max_shift = np.max(np.abs(smoothed[mask_bool] - filled[mask_bool]))
            filled[mask_bool] = smoothed[mask_bool]
            if max_shift < tolerance:
                break
                
        filled = np.clip(filled, 0, 255).astype(np.uint8)
        return Image.fromarray(filled)
