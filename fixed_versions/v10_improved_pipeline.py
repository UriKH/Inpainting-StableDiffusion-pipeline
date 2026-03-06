###############################
# noisy mask - NOVEL!         #
###############################

from v7_improved_pipeline import ImprovedInpaintPipeline
import torchvision.transforms.functional as TF
import torch
import pythonperlin as perlin
import torch.nn.functional as F
import utils.torch_utils as utils

# 1: defaults
# 2: res=8, thres=linear
# 3: res=4, thres=linear
# 4: res=4, thres=linear, kernel=25
# 5: res=8, thres=linear, kernel=15
# 6: res=2, thres=linear
# 7: res=3, thres=linear, kernel=25

class ImprovedInpaintPipelineV10(ImprovedInpaintPipeline):
    def compute_organic_mask(self, base_mask, t, max_t, noise_res=16, dilation_kernel=11, thresh='linear'):
        """
        Generates a time-dependent, organic 'coastline' mask for latent blending.
        """# 1. The Time Ratio (rho) based on actual noise level t
        rho = 1.0 - (t.item() / max_t)

        if t.item() == 0:
            return base_mask
        
        # 3. Dilation: Expand the original mask to give the model room to breathe
        pad = dilation_kernel // 2
        dilated_mask = F.max_pool2d(base_mask, kernel_size=dilation_kernel, stride=1, padding=pad)
        
        smooth_noise = self.generate_perlin_noise_2d(
            shape=base_mask.shape, 
            res=(noise_res, noise_res), 
            device=base_mask.device
        )
        
        # 5. Combine and Apply Dynamic Threshold
        noisy_mask = dilated_mask * smooth_noise
        
        # As time progresses (rho -> 1), the threshold gets higher, 
        # forcing the organic boundaries to shrink back toward the center.
        if thresh == 'linear':
            current_threshold = 0.2 + (rho * 0.6) 
        elif thresh == 'cubic':
            current_threshold = 0.2 + ((rho ** 2) * 0.6)
        else:
            raise ValueError(f'bad threshold type {thresh}. Use: linear/cubic')
        # Binarize back to strict 0s and 1s to create the rocky coastline
        dynamic_mask = (noisy_mask > current_threshold).float()
        
        # 6. Safety Net: The mask must NEVER shrink smaller than the original missing hole
        return torch.max(dynamic_mask, base_mask)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        schedule_indices = self._get_repaint_schedule(num_inference_steps)
        
        noise = torch.randn_like(init_latents)
        latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[schedule_indices[0]]) * (1 - mask_tensor))
                   + (noise * mask_tensor))
        
        _, _, latent_h, latent_w = init_latents.shape
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        # initial injection
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor)
        max_t = self.scheduler.config.num_train_timesteps
        
        try:
            for idx, step_index in enumerate(schedule_indices):
                t = self.scheduler.timesteps[step_index]
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # update injection
                dynamic_mask = self.compute_organic_mask(
                    base_mask=mask_tensor, 
                    t=t, 
                    max_t=max_t,
                    noise_res=3,
                    thresh='linear',
                    dilation_kernel=25
                )

                dynamic_soft_mask = self._create_soft_mask(dynamic_mask)
                for name, proc in self.unet.attn_processors.items():
                    if 'attn1' in name:
                        proc.mask_tensor = dynamic_mask
                    elif 'attn2' in name:
                        proc.mask_tensor = dynamic_soft_mask
    
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
                
                latents = (known_background * (1 - dynamic_mask)) + (latents * dynamic_mask)
        finally:
            self._remove_masked_attention()
        return latents