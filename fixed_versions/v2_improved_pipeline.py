import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2 as cv
from v1_improved_pipeline import ImprovedInpaintPipelineV1
from pipeline import InpaintPipelineInput
from utils import torch_utils as utils


class ImprovedInpaintPipelineV2(ImprovedInpaintPipelineV1):
    def __init__(self, jump_length=10, jump_n_sample=2):
        # Inherit Time-Travel parameters and Negative Prompts from V1
        super().__init__(jump_length=jump_length, jump_n_sample=jump_n_sample)
        self.feather_radius = 10
        self.dilate_kernel_size = 5

    def mask_preprocessing(self, mask_image: Image.Image):
        """
        Enhances the mask using Dilation and Feathering to prevent 'cut' edges.
        """
        # 1. Dilate mask: Expand slightly to give U-Net context inside the boundary
        mask_np = np.array(mask_image.convert("L"))
        kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
        mask_dilated = cv.dilate(mask_np, kernel, iterations=1)
        
        # 2. Feather mask: Use Gaussian blur to create a soft transition gradient
        mask_pil = Image.fromarray(mask_dilated).filter(ImageFilter.GaussianBlur(radius=self.feather_radius))
        return mask_pil

# import torch
# import numpy as np
# from PIL import Image, ImageFilter
# import cv2 as cv
# from vanilla_pipeline import InpaintPipeline
# from pipeline import InpaintPipelineInput

# class ImprovedInpaintPipeline(InpaintPipeline):
#     def __init__(self):
#         # Inherit directly from Vanilla to avoid Time-Travel clashes
#         super().__init__()
#         self.feather_radius = 10
#         self.dilate_kernel_size = 5

#     def preprocess(self, pipe_in: InpaintPipelineInput):
#         """
#         Dilate and Feather the mask for soft boundaries.
#         """
#         mask_np = np.array(pipe_in.mask_image.convert("L"))
        
#         # 1. Dilate
#         kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
#         mask_dilated = cv.dilate(mask_np, kernel, iterations=1)
        
#         # 2. Feather (Gaussian Blur)
#         mask_pil = Image.fromarray(mask_dilated).filter(ImageFilter.GaussianBlur(radius=self.feather_radius))
#         pipe_in.mask_image = mask_pil

#         # Pass back to vanilla to handle image formatting
#         return super().preprocess(pipe_in)
    
#     '''
#     @torch.no_grad()
#     def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
#         """
#         Standard linear denoising loop with Annealed Blending and Energy Normalization.
#         """
#         self.scheduler.set_timesteps(num_inference_steps, device=self.device)
#         noise = torch.randn_like(init_latents)

#         # Initial Setup (mask_tensor: 1=hole, 0=background)
#         latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[0]) * (1 - mask_tensor))
#                    + (noise * mask_tensor))

#         for i, t in enumerate(self.scheduler.timesteps):
#             # 1. U-Net Prediction
#             latent_model_input = torch.cat([latents] * 2)
#             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#             noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
#             # CFG
#             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)

#             # 2. Step the scheduler
#             latents = self.scheduler.step(noise_pred, t, latents).prev_sample

#             # 3. Prepare background for the current step
#             if i < len(self.scheduler.timesteps) - 1:
#                 t_next = self.scheduler.timesteps[i + 1]
#                 background_noise = torch.randn_like(init_latents)
#                 known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
#             else:
#                 known_background = init_latents

#             # --- 4. ANNEALED FEATHERED BLENDING ---
#             progress = i / len(self.scheduler.timesteps)
            
#             # We shrink the generation weight by up to 15% at the very end
#             # This allows the background to bleed into the feathered edges
#             anneal_factor = 1.0 - (0.15 * progress) 
            
#             bg_weight = (1 - mask_tensor)
#             gen_weight = mask_tensor * anneal_factor

#             # Blend
#             latents = (known_background * bg_weight) + (latents * gen_weight)
            
#             # Normalize to prevent the gray/dark noise at the blurred edges
#             latents = latents / (bg_weight + gen_weight + 1e-6)

#         return latents
    
    
#     @torch.no_grad()
#     def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
#         self.scheduler.set_timesteps(num_inference_steps, device=self.device)
#         schedule_indices = self._get_repaint_schedule(num_inference_steps) # From V1
        
#         # --- NEW: Find exactly where Phase 1 ends and Phase 2 begins ---
#         last_jump_idx = -1
#         for i in range(len(schedule_indices) - 1):
#             if schedule_indices[i+1] < schedule_indices[i]: # Identifies backward jumps
#                 last_jump_idx = i
                
#         noise = torch.randn_like(init_latents)
#         latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[schedule_indices[0]]) * (1 - mask_tensor)) 
#                 + (noise * mask_tensor))

#         for idx, step_index in enumerate(schedule_indices):
#             t = self.scheduler.timesteps[step_index]
            
#             # Predict noise
#             latent_model_input = torch.cat([latents] * 2)
#             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
#             noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
#             # CFG
#             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)

#             latents = self.scheduler.step(noise_pred, t, latents).prev_sample

#             # Handle Background Context & Jumps
#             is_last_step = (idx == len(schedule_indices) - 1)
#             if not is_last_step:
#                 next_step_index = schedule_indices[idx + 1]
#                 t_next = self.scheduler.timesteps[next_step_index]
                
#                 # Time-travel! Add noise back.
#                 if next_step_index < step_index: 
#                     beta = self.scheduler.betas[t_next].to(self.device)
#                     latents = torch.sqrt(1 - beta) * latents + torch.sqrt(beta) * torch.randn_like(latents)
                
#                 background_noise = torch.randn_like(init_latents)
#                 known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
#             else:
#                 known_background = init_latents

#             # --- THE SMARTER BLENDING LOGIC ---
#             if idx <= last_jump_idx:
#                 # PHASE 1: Time-Travel Resampling Active
#                 # Use strict blending to avoid corrupting latents before adding noise
#                 latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
#             else:
#                 # PHASE 2: Final Descent
#                 # No more jumps, safe to apply Annealed Feathering
#                 phase2_total_steps = len(schedule_indices) - 1 - last_jump_idx
#                 current_phase2_step = idx - last_jump_idx
                
#                 progress = current_phase2_step / phase2_total_steps if phase2_total_steps > 0 else 1.0
#                 anneal_factor = 1.0 - (0.05 * progress) # Reduce mask strength by up to 20%
                
#                 bg_weight = (1 - mask_tensor)
#                 gen_weight = mask_tensor * anneal_factor
                
#                 # Blend and Normalize to prevent energy loss
#                 latents = (known_background * bg_weight) + (latents * gen_weight)
#                 latents = latents / (bg_weight + gen_weight + 1e-6)

#         return latents

#         '''


#     @torch.no_grad()
#     def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
#         self.scheduler.set_timesteps(num_inference_steps, device=self.device)
#         noise = torch.randn_like(init_latents)

#         # Polarity: 1.0 = Hole (Noise), 0.0 = Background (Keep)
#         latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[0]) * (1 - mask_tensor))
#                    + (noise * mask_tensor))

#         for i, t in enumerate(self.scheduler.timesteps):
#             latent_model_input = torch.cat([latents] * 2)
#             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#             noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#             # CFG
#             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)

#             latents = self.scheduler.step(noise_pred, t, latents).prev_sample

#             # Compute aligned background for the exact same timestep
#             if i < len(self.scheduler.timesteps) - 1:
#                 t_next = self.scheduler.timesteps[i + 1]
#                 background_noise = torch.randn_like(init_latents)
#                 known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
#             else:
#                 known_background = init_latents

#             # --- VARIANCE-PRESERVING ANNEALED BLENDING ---
#             progress = i / len(self.scheduler.timesteps)

#             # The background weight slightly relaxes by up to 20% in the final steps
#             anneal_factor = 1.0 - (0.2 * progress)

#             # Weights based on correct polarity (1 = Hole)
#             w_bg = (1 - mask_tensor) * anneal_factor
#             w_gen = mask_tensor

#             # 1. Blend
#             latents = (known_background * w_bg) + (latents * w_gen)

#             # 2. Normalize Energy (Crucial for blurred masks)
#             # Notice that if w_gen=0 (pure background), variance_norm equals w_bg.
#             # Dividing w_bg by w_bg equals 1.0, preserving the background perfectly!
#             variance_norm = torch.sqrt(w_bg**2 + w_gen**2 + 1e-8)
#             latents = latents / variance_norm

#         return latents
