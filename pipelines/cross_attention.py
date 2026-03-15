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
        self.mask_tensor = None
        self.resize_mode = None
        self.default_processor = AttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        # Run default processor first
        res = self.default_processor(
            attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask, temb=temb, *args, **kwargs
        )

        is_cross_attention = encoder_hidden_states is not None
        if not is_cross_attention or self.mask_tensor is None:
            return res

        B, N, D = res.shape
        ratio = math.sqrt((self.latent_h * self.latent_w) / N)
        h_i = int(self.latent_h / ratio)
        w_i = int(self.latent_w / ratio)

        mask_resized = F.interpolate(self.mask_tensor, size=(h_i, w_i), mode=self.resize_mode)
        mask_flat = mask_resized.view(1, N, 1).to(res.device)

        if B == 2:
            uncond_res, cond_res = res.chunk(2)
            cond_res_blended = cond_res * mask_flat + uncond_res * (1 - mask_flat)
            res = torch.cat([uncond_res, cond_res_blended])
        return res
