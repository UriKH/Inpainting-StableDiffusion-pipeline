from v5_improved_pipeline import MaskedCrossAttnProcessor, ImprovedInpaintPipeline as InpaintPipeline
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

class MaskedSelfAttnProcessor:
    def __init__(self, latent_h, latent_w):
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.mask_tensor = None  # Expected shape: (1, 1, H, W) where 1 is the hole

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        # 1. Standard projection
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 2. Dynamic Spatial Resizing for the mask
        if self.mask_tensor is not None:
            ratio = math.sqrt((self.latent_h * self.latent_w) / sequence_length)
            h_i = int(self.latent_h / ratio)
            w_i = int(self.latent_w / ratio)
            
            # Using nearest to keep the boolean logic clean for self-attention
            mask_resized = F.interpolate(self.mask_tensor, size=(h_i, w_i), mode='nearest')
            mask_flat = mask_resized.view(1, sequence_length)  # Shape: (1, N)
            
            # 3. Build the Bias Matrix B
            # Q_mask shape: (1, N, 1) -> Represents the queries
            # K_mask shape: (1, 1, N) -> Represents the keys
            Q_mask = mask_flat.unsqueeze(-1)
            K_mask = mask_flat.unsqueeze(-2)
            
            # Rule: If Query is Background (0) and Key is Hole (1), Forbid it.
            forbidden = (Q_mask == 0) & (K_mask == 1)
            
            # Create the bias tensor and fill forbidden connections with -10000
            bias = torch.zeros((1, 1, sequence_length, sequence_length), device=query.device, dtype=query.dtype)
            bias.masked_fill_(forbidden, -10000.0)
            
            # Expand to match batch size (for CFG) and heads
            bias = bias.expand(batch_size, attn.heads, sequence_length, sequence_length)
            
            # Combine with existing attention_mask if one was passed by the UNet
            if attention_mask is not None:
                attention_mask = attention_mask + bias
            else:
                attention_mask = bias

        # 4. Calculate Attention using PyTorch's optimized function
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self, jump_length=10, jump_n_sample=2):
        super().__init__()
        # Store resampling hyperparameters as instance variables 
        # to preserve the base pipe() signature.
        self.jump_length = jump_length
        self.jump_n_sample = jump_n_sample
    
    def _inject_masked_attention(self, latent_h, latent_w, cross_mask, self_mask):
        """Injects custom processors into the UNet."""
        processor_dict = {}
        for name in self.unet.attn_processors.keys():
            if "attn1" in name:  # Self-Attention Layers
                processor = MaskedSelfAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = self_mask
                processor_dict[name] = processor
            elif "attn2" in name:  # Cross-Attention Layers
                processor = MaskedCrossAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = cross_mask
                processor_dict[name] = processor
            else:
                processor_dict[name] = AttnProcessor2_0()
                
        self.unet.set_attn_processor(processor_dict)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        """Overrides the base denoise method to include time-travel resampling."""
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 1. Generate the custom time-travel schedule
        schedule_indices = self._get_repaint_schedule(num_inference_steps)
        
        # 2. Initial Setup
        noise = torch.randn_like(init_latents)
        latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[schedule_indices[0]]) * (1 - mask_tensor))
                   + (noise * mask_tensor))
        
        _, _, latent_h, latent_w = init_latents.shape
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor)
        
        try:
            for idx, step_index in enumerate(schedule_indices):
                t = self.scheduler.timesteps[step_index]
    
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                # Predict noise
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)
    
                # Step the scheduler to get t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # 3. Handle Time-Travel and Background Context
                is_last_step = (idx == len(schedule_indices) - 1)
            
                if not is_last_step:
                    next_step_index = schedule_indices[idx + 1]
                    t_next = self.scheduler.timesteps[next_step_index]
                
                    is_jump_backward = next_step_index < step_index
                
                    if is_jump_backward:
                        # Time travel! Apply the DDPM forward equation to inject noise back in
                        beta = self.scheduler.betas[t_next].to(self.device)
                        noise = torch.randn_like(latents)
                        latents = torch.sqrt(1 - beta) * latents + torch.sqrt(beta) * noise
                    
                    # Generate fresh stochastic background context for the upcoming step
                    background_noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
                else:
                    # Final cleanup step resolving to sharpness
                    known_background = init_latents
    
                # 4. Blend the accurately aligned latents
                latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
        finally:
            self._remove_masked_attention()
        return latents
