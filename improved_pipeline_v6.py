from vanilla_pipeline import InpaintPipeline
import torch
import numpy as np
import cv2 as cv
from PIL import Image
import torchvision.transforms.functional as TF


# 1. Strict Binary Dilation Mask (No blur here, we want strict latent boundaries!)
def mask_op(image):
    mask_np = np.array(image)
    _, mask_np = cv.threshold(mask_np, 127, 255, cv.THRESH_BINARY)
    
    # Large dilation to give the AI a canvas to cast shadows
    dilate_kernel = np.ones((15, 15), np.uint8)
    mask_np = cv.dilate(mask_np, dilate_kernel, iterations=1)
    
    return Image.fromarray(mask_np).resize((64, 64), Image.LANCZOS)

class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self):
        super().__init__()
        # Override the mask operation specifically for the improved pipeline
        self.__class__.MASK_DEFAULT_PREPROC_OP = staticmethod(mask_op)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = torch.randn_like(init_latents)

        # 2. Strict Vanilla Latent Blending
        # This guarantees the background (onions) stays safe, while the AI draws the cupcake
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

            # --- ANNEALED MASK LOGIC ---
            # Calculate how much to blur based on how far along we are in the loop.
            # It starts at sigma=5.0 (very blurry) and drops to sigma=0.1 (basically binary)
            progress = i / len(self.scheduler.timesteps)
            current_sigma = max(0.1, 2.0 * (1.0 - progress))
            
            # Apply the dynamic blur to the base mask tensor
            # We use an odd kernel size (15) and our decaying sigma
            dynamic_mask = TF.gaussian_blur(mask_tensor, kernel_size=5, sigma=current_sigma)

            # Blend using the dynamic mask!
            noise = torch.randn_like(init_latents)
            noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t_next)
            latents = (noisy_init_latents * (1 - dynamic_mask)) + (latents * dynamic_mask)

        return latents

    @torch.no_grad()
    def pipe(self, pipe_in):
        # We need a 512x512 mask for math. Our mask_op downscaled it to 64x64, so we scale it back up purely for finding the bounding box.
        mask_64 = pipe_in.mask_image
        mask_512 = mask_64.resize((512, 512), Image.NEAREST)
        mask_cv = np.array(mask_512)
        
        # ==========================================
        # 1. SMART CROP: Find the Bounding Box
        # ==========================================
        y_indices, x_indices = np.where(mask_cv > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            mask_w = x_max - x_min
            mask_h = y_max - y_min
            
            # Create a square crop with 50% padding around the mask so the AI has context
            side = int(max(mask_w, mask_h) * 1.5)
            side = max(side, 128)  # Don't crop too small
            side = min(side, 512)  # Don't crop bigger than the image
            
            # Calculate the top-left corner of the crop
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            
            # Keep it inside the image bounds
            if x1 + side > 512: x1 = 512 - side
            if y1 + side > 512: y1 = 512 - side
        else:
            x1, y1, side = 0, 0, 512 # Fallback if mask is empty
            
        # ==========================================
        # 2. ZOOM IN: Prepare the cropped inputs
        # ==========================================
        crop_box = (x1, y1, x1 + side, y1 + side)
        cropped_init = pipe_in.init_image.crop(crop_box).resize((512, 512), Image.LANCZOS)
        cropped_mask = mask_512.crop(crop_box).resize((64, 64), Image.NEAREST)
        
        # Manually encode these zoomed-in crops to latents
        image_np = np.array(cropped_init).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        init_latents = self.SD_SCALE_FACTOR * self.vae.encode(image_tensor).latent_dist.sample()
        
        mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # ==========================================
        # 3. GENERATE: Run the AI on the zoomed-in space!
        # ==========================================
        text_embeddings = self.encode_prompt(pipe_in.prompt, self.text_encoder, self.tokenizer)
        latents = self.denoise(text_embeddings, init_latents, mask_tensor)
        ai_generated_crop = self.decode_latents(latents)

        # ==========================================
        # 4. RESTORE: Shrink and paste it back perfectly
        # ==========================================
        # Shrink the generated object back down to its actual tiny size
        ai_generated_crop_restored = ai_generated_crop.resize((side, side), Image.LANCZOS)
        
        # Paste it exactly where it belongs on the original background
        full_ai_image = pipe_in.init_image.copy()
        full_ai_image.paste(ai_generated_crop_restored, (x1, y1))

        # ==========================================
        # 5. POISSON BLEND: Fix the lighting seams
        # ==========================================
        src = np.array(full_ai_image)
        dst = np.array(pipe_in.init_image)
        
        if len(mask_cv.shape) == 2:
            mask_cv = cv.cvtColor(mask_cv, cv.COLOR_GRAY2RGB)
            
        x, y, w, h = cv.boundingRect(mask_cv[:, :, 0])
        if w == 0 or h == 0:
            return full_ai_image
            
        center = (x + w // 2, y + h // 2)
        blended_np = cv.seamlessClone(src, dst, mask_cv, center, cv.NORMAL_CLONE)
        
        return Image.fromarray(blended_np)
