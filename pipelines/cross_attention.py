from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import math


class MaskedCrossAttnProcessor:
    def __init__(self, latent_h, latent_w):
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.mask_tensor = None  # Will be dynamically injected
        self.dilation_threshold = 0.0
        self.default_processor = AttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        # 1. Run the standard optimized attention calculation
        res = self.default_processor(
            attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask, temb=temb, *args, **kwargs
        )

        # 2. Safety check: Only apply masking if it is Cross-Attention and we have a mask
        is_cross_attention = encoder_hidden_states is not None
        if not is_cross_attention or self.mask_tensor is None:
            return res

        # 3. Dynamic Spatial Resizing
        B, N, D = res.shape
        # Calculate current spatial dimensions based on sequence length (N = H * W)
        ratio = math.sqrt((self.latent_h * self.latent_w) / N)
        h_i = int(self.latent_h / ratio)
        w_i = int(self.latent_w / ratio)

        # Interpolate the base mask to this layer's exact resolution
        # mask_resized = F.interpolate(self.mask_tensor, size=(h_i, w_i), mode='nearest')

        mask_resized = F.interpolate(self.mask_tensor, size=(h_i, w_i), mode='area')
        mask_flat = (mask_resized > self.dilation_threshold).view(1, N, 1).to(res.device)

        # Flatten to match the sequence length: shape (1, N, 1)
        # mask_flat = mask_resized.view(1, N, 1).to(res.device)

        # 4. Apply the Regional Prompting Mathematics
        # CFG passes a batch of 2 (uncond, cond)
        if B == 2:
            uncond_res, cond_res = res.chunk(2)
            # Blend: Use conditional context inside the mask, unconditional outside
            cond_res_blended = cond_res * mask_flat + uncond_res * (1 - mask_flat)
            res = torch.cat([uncond_res, cond_res_blended])

        return res
