######################################### 
# Attention shit, resampling, FreeU
#########################################
from v7_improved_pipeline import ImprovedInpaintPipeline as InpaintPipeline7
from pipeline import InpaintPipelineInput
import torch
from PIL import Image
import cv2 as cv
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
import math
import torchvision.transforms.functional as TF

import torch
import torch.nn.functional as F
import math


#self.apply_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
#self.apply_freeu(s1=0.95, s2=0.8, b1=1.1, b2=1.1)

# first: all 1's
# 1: s1=1, s2=1, b1=1.05, b2=1.05
# 2: s1=1, s2=1, b1=1, b2=1.05
# 3: s1=0.95, s2=0.95, b1=1, b2=1.05
# 4: s1=0.95, s2=0.95, b1=1.05, b2=1.05
# 5: s1=0.95, s2=0.95, b1=1, b2=1
# 7: s1=0.98, s2=0.98, b1=1.05, b2=1.05
# 8: s1=1, s2=1, b1=1.03, b2=1.03
# 8: s1=1, s2=1, b1=1.07, b2=1.07

class ImprovedInpaintPipeline(InpaintPipeline7):
    def __init__(self, jump_length=10, jump_n_sample=2):
        super().__init__()
        # Store resampling hyperparameters as instance variables 
        # to preserve the base pipe() signature.
        self.jump_length = jump_length
        self.jump_n_sample = jump_n_sample
    
    def apply_freeu(self, s1=0.9, s2=0.2, b1=1.2, b2=1.4):
        """
        Modifies the UNet activations during the forward pass to prioritize 
        semantic structure over high-frequency background matching.
        """
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def remove_freeu(self):
        """Restores standard UNet architecture scaling."""
        self.unet.disable_freeu()
    
    def pipe(self, pipe_in: InpaintPipelineInput):
        self.apply_freeu(s1=0.95, s2=0.95, b1=1, b2=1)
        res = super().pipe(pipe_in)
        self.remove_freeu()
        return res
