###############################
# solve edge - mask blending  #
###############################

from v7_improved_pipeline import ImprovedInpaintPipeline
import torchvision.transforms.functional as TF
import torch


class ImprovedInpaintPipelineV9(ImprovedInpaintPipeline):
    def _get_dynamic_mask(self, mask_tensor, current_idx, total_steps, max_sigma=5.0):
        """Linearly anneals the mask blur from max_sigma down to a hard boundary."""
        progress = current_idx / max(1, (total_steps - 1))
        # Decay sigma as we get closer to step 0
        current_sigma = max(0.01, max_sigma * (1.0 - progress))
        
        if current_sigma <= 0.5:
            return mask_tensor # Return hard mask at the end for perfect background
            
        return TF.gaussian_blur(mask_tensor, kernel_size=[7, 7], sigma=[current_sigma, current_sigma])

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        schedule_indices = self._get_repaint_schedule(num_inference_steps)
        
        noise = torch.randn_like(init_latents)
        latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[schedule_indices[0]]) * (1 - mask_tensor))
                   + (noise * mask_tensor))
        
        _, _, latent_h, latent_w = init_latents.shape
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor)
        
        try:
            for idx, step_index in enumerate(schedule_indices):
                t = self.scheduler.timesteps[step_index]
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                is_last_step = (idx == len(schedule_indices) - 1)
                if not is_last_step:
                    next_step_index = schedule_indices[idx + 1]
                    t_next = self.scheduler.timesteps[next_step_index]
                    if next_step_index < step_index:
                        beta = self.scheduler.betas[t_next].to(self.device)
                        noise = torch.randn_like(latents)
                        latents = torch.sqrt(1 - beta) * latents + torch.sqrt(beta) * noise
                    
                    background_noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
                else:
                    known_background = init_latents
    
                # ZONE 1 INTERVENTION: Dynamically soft mask blending
                dynamic_mask = self._get_dynamic_mask(mask_tensor, idx, len(schedule_indices))
                latents = (known_background * (1 - dynamic_mask)) + (latents * dynamic_mask)
        finally:
            self._remove_masked_attention()
        return latents