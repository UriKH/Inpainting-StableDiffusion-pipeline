from old.vanilla_pipeline import InpaintPipeline
import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from PIL import Image
import math

# ==========================================
# 1. THE CUSTOM ATTENTION PROCESSOR
# ==========================================
class MaskedSelfAttentionProcessor:
    def __init__(self, base_mask_tensor):
        # The mask tensor from our Smart Crop (shape: [1, 1, 64, 64])
        self.base_mask = base_mask_tensor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        # We only want to hack Image-to-Image (Self-Attention). 
        # If text conditioning (encoder_hidden_states) is present, we process it normally.
        is_cross_attention = encoder_hidden_states is not None
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        if is_cross_attention:
            encoder_hidden_states = encoder_hidden_states
        else:
            encoder_hidden_states = hidden_states

        # Extract Query, Key, Value
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape for multi-head attention: (batch, heads, sequence_length, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Calculate standard attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)

        # --- THE HACK: Apply Mask Bias ONLY for Self-Attention ---
        if not is_cross_attention:
            # Find the resolution of this specific layer (e.g., 64x64, 32x32, 16x16)
            spatial_size = int(math.sqrt(sequence_length))
            
            # Dynamically resize our mask to match the layer's internal resolution
            current_mask = F.interpolate(
                self.base_mask.to(dtype=attention_scores.dtype, device=attention_scores.device),
                size=(spatial_size, spatial_size),
                mode='nearest'
            )
            
            # Flatten to (1, 1, sequence_length)
            flat_mask = current_mask.view(1, 1, sequence_length)
            
            # query_mask represents the pixels "looking"
            # key_mask represents the pixels "being looked at"
            query_mask = flat_mask.unsqueeze(-1)
            key_mask = flat_mask.unsqueeze(-2)
            
            # Rule: Background (0) cannot look at Foreground (1).
            forbidden_connections = (1.0 - query_mask) * key_mask
            
            # Sever the connections by applying a massive mathematical penalty
            bias = forbidden_connections * -10000.0
            attention_scores = attention_scores + bias

        # Softmax turns the -10000.0 penalty into 0.0 attention!
        attention_probs = attention_scores.softmax(dim=-1)
        
        # Finish the standard attention math
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)

        return hidden_states


# ==========================================
# 2. PIPELINE HELPERS
# ==========================================
def mask_op(image):
    mask_np = np.array(image)
    _, mask_np = cv.threshold(mask_np, 127, 255, cv.THRESH_BINARY)
    # Standard dilation buffer
    dilate_kernel = np.ones((25, 25), np.uint8)
    mask_np = cv.dilate(mask_np, dilate_kernel, iterations=1)
    return Image.fromarray(mask_np).resize((64, 64), Image.NEAREST)


class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self):
        super().__init__()
        self.__class__.MASK_DEFAULT_PREPROC_OP = staticmethod(mask_op)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = torch.randn_like(init_latents)

        # Vanilla Latent Blending loop
        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if i < len(self.scheduler.timesteps) - 1:
                t_next = self.scheduler.timesteps[i + 1]
            else:
                t_next = torch.tensor([0], device=self.device)

            noise = torch.randn_like(init_latents)
            noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t_next)
            
            latents = (noisy_init_latents * (1 - mask_tensor)) + (latents * mask_tensor)
            
        return latents

    @torch.no_grad()
    def pipe(self, pipe_in):
        mask_64 = pipe_in.mask_image
        mask_512 = mask_64.resize((512, 512), Image.NEAREST)
        mask_cv = np.array(mask_512)

        if len(mask_cv.shape) == 3:
            mask_cv_gray = cv.cvtColor(mask_cv, cv.COLOR_RGB2GRAY)
        else:
            mask_cv_gray = mask_cv

        contours, _ = cv.findContours(mask_cv_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return pipe_in.init_image.copy()

        current_bg = pipe_in.init_image.copy()

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w < 5 or h < 5:
                continue

            side = int(max(w, h) * 1.5)
            side = max(side, 128)
            side = min(side, 512)

            cx, cy = x + w // 2, y + h // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)

            if x1 + side > 512: x1 = 512 - side
            if y1 + side > 512: y1 = 512 - side
            crop_box = (x1, y1, x1 + side, y1 + side)

            local_mask_cv = np.zeros_like(mask_cv_gray)
            cv.drawContours(local_mask_cv, [contour], -1, 255, thickness=cv.FILLED)
            local_mask_pil = Image.fromarray(local_mask_cv)
            
            cropped_init = current_bg.crop(crop_box).resize((512, 512), Image.LANCZOS)
            cropped_mask = local_mask_pil.crop(crop_box).resize((64, 64), Image.NEAREST)

            image_np = np.array(cropped_init).astype(np.float32) / 127.5 - 1.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            init_latents = self.SD_SCALE_FACTOR * self.vae.encode(image_tensor).latent_dist.sample()

            mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)

            # ==========================================
            # 3. INJECT THE MASKED ATTENTION PROCESSOR
            # ==========================================
            # We iterate through the UNet's layers and inject our custom class ONLY into the Self-Attention blocks ('attn1')
            new_processors = {}
            for name, processor in self.unet.attn_processors.items():
                if name.endswith("attn1.processor"):
                    new_processors[name] = MaskedSelfAttentionProcessor(mask_tensor)
                else:
                    new_processors[name] = processor # Keep cross-attention default
            
            self.unet.set_attn_processor(new_processors)

            # Generate! The UNet is now mathematically blocked from looking at noise.
            text_embeddings = self.encode_prompt(pipe_in.prompt, self.text_encoder, self.tokenizer)
            latents = self.denoise(text_embeddings, init_latents, mask_tensor)
            ai_generated_crop = self.decode_latents(latents)

            # ==========================================
            # 4. RESTORE AND CLEANUP
            # ==========================================
            orig_crop_np = np.array(cropped_init)
            ai_crop_np = np.array(ai_generated_crop)
            
            diff = cv.absdiff(orig_crop_np, ai_crop_np)
            diff_gray = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)
            _, dynamic_mask_highres = cv.threshold(diff_gray, 15, 255, cv.THRESH_BINARY)
            dynamic_mask_highres = cv.morphologyEx(dynamic_mask_highres, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
            dynamic_mask_highres = cv.dilate(dynamic_mask_highres, np.ones((9, 9), np.uint8), iterations=1)

            ai_generated_crop_restored = ai_generated_crop.resize((side, side), Image.LANCZOS)
            dynamic_mask_small = Image.fromarray(dynamic_mask_highres).resize((side, side), Image.NEAREST)
            dynamic_mask_small_cv = np.array(dynamic_mask_small)

            full_ai_image = current_bg.copy()
            full_ai_image.paste(ai_generated_crop_restored, (x1, y1))

            final_clone_mask = np.zeros((512, 512), dtype=np.uint8)
            final_clone_mask[y1:y1+side, x1:x1+side] = dynamic_mask_small_cv
            
            src = np.array(full_ai_image)
            dst = np.array(current_bg)
            final_clone_mask_color = cv.cvtColor(final_clone_mask, cv.COLOR_GRAY2RGB)
            
            x_d, y_d, w_d, h_d = cv.boundingRect(final_clone_mask)
            if w_d > 0 and h_d > 0:
                center = (x_d + w_d // 2, y_d + h_d // 2)
                try:
                    blended_np = cv.seamlessClone(src, dst, final_clone_mask_color, center, cv.NORMAL_CLONE)
                    current_bg = Image.fromarray(blended_np)
                except Exception:
                    current_bg = full_ai_image

        return current_bg
