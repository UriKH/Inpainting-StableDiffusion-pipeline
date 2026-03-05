from v3_improved_pipeline import ImprovedInpaintPipeline as InpaintPipeline
from pipeline import InpaintPipelineInput
import torch
from PIL import Image
import cv2 as cv
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
import math


class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self, jump_length=10, jump_n_sample=2):
        super().__init__(jump_length, jump_n_sample)

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
        self.apply_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        res = super().pipe(pipe_in)
        self.remove_freeu()
        return res
        