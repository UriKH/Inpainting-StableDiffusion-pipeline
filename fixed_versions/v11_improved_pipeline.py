###############################
# latent boundry filter       #
###############################

from v7_improved_pipeline import ImprovedInpaintPipeline
import torchvision.transforms.functional as TF
import torch
import pythonperlin as perlin
import torch.nn.functional as F
import utils.torch_utils as utils


class ImprovedInpaintPipelineV11(ImprovedInpaintPipeline):
    def apply_latent_boundary_filter(self, latents, mask_tensor, t, max_t, blur_kernel=3, base_sigma=1.5):
        """
        Applies a targeted low-pass filter strictly to the boundary of the blended latents.
        """
        # 1. Temporal Annealing: Blur is strongest at high noise (t=1000) and vanishes at t=0
        rho = t.item() / max_t
        if rho == 0:
            return latents
            
        # Dynamically scale the sigma. Add a tiny epsilon (0.1) to prevent PyTorch crash on zero sigma
        current_sigma = (base_sigma * rho) + 0.1 

        # 2. Extract the Boundary Ring
        # A 3x3 kernel creates a 1-pixel thick latent ring (which is an 8-pixel thick ring in the final image)
        dilated = F.max_pool2d(mask_tensor, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask_tensor, kernel_size=3, stride=1, padding=1)
        boundary_mask = dilated - eroded

        # 3. Apply the Low-Pass Filter to smooth the latent manifold
        blurred_latents = TF.gaussian_blur(
            latents, 
            kernel_size=[blur_kernel, blur_kernel], 
            sigma=[current_sigma, current_sigma]
        )

        # 4. Targeted Splicing: Only apply the blurred bridge at the boundary
        filtered_latents = (blurred_latents * boundary_mask) + (latents * (1 - boundary_mask))

        return filtered_latents
    
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

                latents = self.apply_latent_boundary_filter(
                    latents=latents, 
                    mask_tensor=mask_tensor, 
                    t=t, 
                    max_t=max_t,
                    blur_kernel=3, # Keep this small! 3x3 in latent space is 24x24 in pixel space
                    base_sigma=1.5
                )
        finally:
            self._remove_masked_attention()
        return latents
