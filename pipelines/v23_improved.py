from pipelines.vanilla_pipeline import InpaintPipelineVanilla
import torch
from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt


class ImprovedInpaintPipelineV23(InpaintPipelineVanilla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def image_preprocessing(self, real_image, mask_image):
        real_arr = np.array(real_image).astype(np.float32)
        mask_arr = np.array(mask_image)
        
        # True inside the hole, False in the known background
        mask_bool = mask_arr == 255
        
        # 1. Initialize the hole with the mean background color to speed up math
        mean_bg_color = np.mean(real_arr[~mask_bool], axis=0)
        filled_arr = real_arr.copy()
        filled_arr[mask_bool] = mean_bg_color
        
        # 2. The Jacobi Stencil (Average of North, South, East, West)
        # Center is 0, edges are 0.25
        kernel = np.array([[0.0,  0.25, 0.0],
                           [0.25, 0.0,  0.25],
                           [0.0,  0.25, 0.0]], dtype=np.float32)
        
        # 3. Iterate the PDE solver
        max_iters = 10000     # A huge safety net
        tolerance = 0.1

        for i in range(max_iters):
            smoothed = cv.filter2D(filled_arr, -1, kernel)
            
            # Calculate the mathematical difference between this step and the last step
            # We only care about the change INSIDE the hole
            max_shift = np.max(np.abs(smoothed[mask_bool] - filled_arr[mask_bool]))
            
            filled_arr[mask_bool] = smoothed[mask_bool]
            
            # The convergence check
            if max_shift < tolerance:
                break
                
        filled_arr = np.clip(filled_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(filled_arr)
