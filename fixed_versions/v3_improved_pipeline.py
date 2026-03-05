from vanilla_pipeline import InpaintPipeline, InpaintPipelineInput
import torch
from PIL import Image
import cv2 as cv
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
import math

class MaskedCrossAttnProcessor:
    def __init__(self, latent_h, latent_w):
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.mask_tensor = None  # Will be dynamically injected
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
        mask_resized = F.interpolate(self.mask_tensor, size=(h_i, w_i), mode='nearest')

        # Flatten to match the sequence length: shape (1, N, 1)
        mask_flat = mask_resized.view(1, N, 1).to(res.device)

        # 4. Apply the Regional Prompting Mathematics
        # CFG passes a batch of 2 (uncond, cond)
        if B == 2:
            uncond_res, cond_res = res.chunk(2)
            # Blend: Use conditional context inside the mask, unconditional outside
            cond_res_blended = cond_res * mask_flat + uncond_res * (1 - mask_flat)
            res = torch.cat([uncond_res, cond_res_blended])

        return res


class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self, jump_length=10, jump_n_sample=2):
        super().__init__()
        # Store resampling hyperparameters as instance variables 
        # to preserve the base pipe() signature.
        self.jump_length = jump_length
        self.jump_n_sample = jump_n_sample
    
    def _inject_masked_attention(self, latent_h, latent_w, mask_tensor):
        """Replaces standard cross-attention with our Masked processor."""
        processor_dict = {}
        for name in self.unet.attn_processors.keys():
            if "attn2" in name:  # The standard naming convention for cross-attention
                processor = MaskedCrossAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = mask_tensor
                processor_dict[name] = processor
            else:
                processor_dict[name] = AttnProcessor2_0() # Keep self-attention default

        self.unet.set_attn_processor(processor_dict)

    def _remove_masked_attention(self):
        """Restores the UNet to its vanilla state to prevent side effects."""
        processor_dict = {
            name: AttnProcessor2_0() for name in self.unet.attn_processors.keys()
        }
        self.unet.set_attn_processor(processor_dict)

    def encode_prompt(self, prompt, text_encoder, tokenizer):
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = tokenizer(
            [
                "ugly, tiling, poorly drawn, out of frame, deformed, blurry, bad anatomy, bad proportions, extra limbs,"
                " artifacts, miniature scene, entire picture, out of context, mismatched lighting"
            ],
            padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def _get_repaint_schedule(self, num_inference_steps):
        """Generates the RePaint sequence of timestep indices."""
        times = list(range(num_inference_steps))
        schedule_indices = []

        i = 0
        jumps_done = 0
        while i < len(times):
            schedule_indices.append(i)
            # Check if we need to jump back in time
            if (i + 1) % self.jump_length == 0 and jumps_done < self.jump_n_sample - 1:
                i = i - self.jump_length + 1
                jumps_done += 1
            else:
                # Reset jump counter and move forward
                if (i + 1) % self.jump_length == 0:
                    jumps_done = 0
                i += 1

        return schedule_indices

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
        self._inject_masked_attention(latent_h, latent_w, mask_tensor)
        
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
