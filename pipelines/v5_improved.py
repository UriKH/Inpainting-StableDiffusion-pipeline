from pipelines.cross_attention import MaskedCrossAttnProcessor
from pipelines.self_attention import MaskedSelfAttnProcessor
from pipelines.v4_improved import ImprovedInpaintPipelineV4
import torch


class ImprovedInpaintPipelineV5(ImprovedInpaintPipelineV4):   
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
            for i, t in enumerate(self.scheduler.timesteps):
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict noise
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)

                # Step the scheduler
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # Update timesteps and add noise
                if i < len(self.scheduler.timesteps) - 1:
                    t_next = self.scheduler.timesteps[i + 1]

                    # Add noise to the original image matching the level we JUST stepped to
                    noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, noise, t_next)
                else:
                    # At the final step, the known background should be perfectly clean
                    known_background = init_latents

                # Blend the accurately aligned latents
                latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
        finally:
            self._remove_masked_attention()
        return latents
