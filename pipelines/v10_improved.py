from pipelines.v9_improved import ImprovedInpaintPipelineV9
import torch
import math


class ImprovedInpaintPipelineV10(ImprovedInpaintPipelineV9):
    def __init__(self,
                 rp_jump_length=10, rp_jump_n_sample=2,
                 ds_min_jumps=1, ds_min_jump_len=5,
                 ds_max_jumps=4, ds_max_jump_len=10,
                 use_dynamic_schedule=False,
                 **kwargs):
        """
        :param rp_jump_length: The length of the jump in the RePaint schedule.
        :param rp_jump_n_sample: The number of samples to jump back in time.
        :param ds_min_jumps: The minimum number of jumps in the dynamic schedule.
        :param ds_min_jump_len: The minimum length of jumps in the dynamic schedule.
        :param ds_max_jumps: The maximum number of jumps in the dynamic schedule.
        :param ds_max_jump_len: The maximum length of jumps in the dynamic schedule.
        :param use_dynamic_schedule: Whether to use the dynamic schedule or the RePaint schedule.
        """
        super().__init__(**kwargs)
        self.jump_length = rp_jump_length
        self.jump_n_sample = rp_jump_n_sample
        self.ds_min_jumps = ds_min_jumps
        self.ds_min_jump_len = ds_min_jump_len
        self.ds_max_jumps = ds_max_jumps
        self.ds_max_jump_len = ds_max_jump_len
        self.use_dynamic_schedule = use_dynamic_schedule
    
    def _get_repaint_schedule(self, num_inference_steps: int) -> list[int]:
        """
        Generates the RePaint sequence of timestep indices.
        :param num_inference_steps: The number of inference steps.
        :return: A list of timestep indices.
        """
        schedule_indices = []

        i = 0
        jumps_done = 0
        while i < num_inference_steps:
            schedule_indices.append(i)

            if (i + 1) % self.jump_length == 0 and jumps_done < self.jump_n_sample - 1:
                i = i - self.jump_length + 1
                jumps_done += 1
            else:
                if (i + 1) % self.jump_length == 0:
                    jumps_done = 0
                i += 1
        return schedule_indices

    def _get_dynamic_schedule(self, num_inference_steps: int) -> list[int]:
        """
        Generates the dynamic sequence of timestep indices where the number of jumps and length decreases over time.
        :param num_inference_steps: The number of inference steps.
        :return: A list of timestep indices.
        """
        schedule_indices = []
        i = 0

        while i < num_inference_steps:
            progress = i / num_inference_steps
            curr_jump_length = int(math.cos(progress * math.pi / 2.) * (self.ds_max_jump_len - self.ds_min_jump_len) +  self.ds_min_jump_len)
            curr_jumps = int(math.cos(progress * math.pi / 2.) * (self.ds_max_jumps - self.ds_min_jumps) + self.ds_min_jumps)

            slice_len = min(curr_jump_length, num_inference_steps - i)
            schedule_indices.extend([i + v for v in range(slice_len)] * curr_jumps)
            i += curr_jump_length
        return schedule_indices

    @torch.no_grad()
    def _initialize_denoise_loop(self, init_latents, mask_tensor, num_inference_steps):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        noise = torch.randn_like(init_latents)

        timesteps = self.scheduler.timesteps
        if self.reconstruction:
            init_step = min(int(num_inference_steps * (1 - self.init_noise_strength)), num_inference_steps - 1)
            timesteps = self.scheduler.timesteps[init_step:]

        if self.use_dynamic_schedule:
            schedule_indices = self._get_dynamic_schedule(len(timesteps))
        else:
            schedule_indices = self._get_repaint_schedule(len(timesteps))

        if self.reconstruction:
            latents = self.scheduler.add_noise(init_latents, noise, timesteps[schedule_indices[0]])
        else:
            latents = ((self.scheduler.add_noise(init_latents, noise, timesteps[schedule_indices[0]])
                        * (1 - mask_tensor)) + (noise * mask_tensor))
        return latents, timesteps, schedule_indices

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        latents, timesteps, schedule_indices = self._initialize_denoise_loop(init_latents, mask_tensor, num_inference_steps)
        _, _, latent_h, latent_w = init_latents.shape

        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor if not self.use_sm_in_sa else soft_attn_mask)
        
        try:
            for idx, step_index in enumerate(schedule_indices):
                t = timesteps[step_index]
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if idx != len(schedule_indices) - 1:
                    next_step_index = schedule_indices[idx + 1]
                    t_next = timesteps[next_step_index]
                
                    if next_step_index < step_index:
                        if step_index + 1 < len(timesteps):
                            t_prev = timesteps[step_index + 1]
                            alpha_prod_prev = self.scheduler.alphas_cumprod[t_prev].to(self.device)
                        else:
                            alpha_prod_prev = torch.tensor(1.0, device=self.device)
                            
                        alpha_prod_target = self.scheduler.alphas_cumprod[t_next].to(self.device)
                        ratio = alpha_prod_target / alpha_prod_prev
                        noise = torch.randn_like(latents)
                        latents = torch.sqrt(ratio) * latents + torch.sqrt(1 - ratio) * noise
                    
                    background_noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
                else:
                    known_background = init_latents
                latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
        finally:
            self._remove_masked_attention()
        return latents
