from old.vanilla_pipeline import InpaintPipeline
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
        # We need a 512x512 mask for math. Our mask_op downscaled it to 64x64.
        mask_64 = pipe_in.mask_image
        mask_512 = mask_64.resize((512, 512), Image.NEAREST)
        mask_cv = np.array(mask_512)

        # 1. Prepare mask for contour detection (handles multiple separate masks!)
        if len(mask_cv.shape) == 3:
            mask_cv_gray = cv.cvtColor(mask_cv, cv.COLOR_RGB2GRAY)
        else:
            mask_cv_gray = mask_cv

        # Find all distinct masked blobs in the image
        contours, _ = cv.findContours(mask_cv_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # If the user passed a completely black mask, just return the original image
        if not contours:
            return pipe_in.init_image.copy()

        # This will hold our rolling updates as we process one mask at a time
        current_bg = pipe_in.init_image.copy()

        # Process each disconnected mask individually
        for contour in contours:
            # Find the bounding box for THIS specific mask blob
            x, y, w, h = cv.boundingRect(contour)

            # 2. FILTER NOISE: Skip tiny accidental brush strokes
            if w < 5 or h < 5:
                continue

            # 3. SMART CROP: Calculate a square box with 50% padding
            side = int(max(w, h) * 1.5)
            side = max(side, 128)  # Don't crop too small
            side = min(side, 512)  # Don't crop bigger than the image

            # Calculate the top-left corner of the crop
            cx, cy = x + w // 2, y + h // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)

            # Keep it strictly inside the image bounds
            if x1 + side > 512: x1 = 512 - side
            if y1 + side > 512: y1 = 512 - side

            crop_box = (x1, y1, x1 + side, y1 + side)

            # 4. ISOLATE MASK & ZOOM IN: Create a temporary mask that ONLY contains this specific shape
            local_mask_cv = np.zeros_like(mask_cv_gray)
            cv.drawContours(local_mask_cv, [contour], -1, 255, thickness=cv.FILLED)
            local_mask_pil = Image.fromarray(local_mask_cv)

            # Zoom in the background and the isolated mask
            cropped_init = current_bg.crop(crop_box).resize((512, 512), Image.LANCZOS)
            cropped_mask = local_mask_pil.crop(crop_box).resize((64, 64), Image.NEAREST)

            # Manually encode these zoomed-in crops to latents
            image_np = np.array(cropped_init).astype(np.float32) / 127.5 - 1.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            init_latents = self.SD_SCALE_FACTOR * self.vae.encode(image_tensor).latent_dist.sample()

            mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)

            # 5. GENERATE: Run the AI on the zoomed-in space!
            text_embeddings = self.encode_prompt(pipe_in.prompt, self.text_encoder, self.tokenizer)
            latents = self.denoise(text_embeddings, init_latents, mask_tensor)
            ai_generated_crop = self.decode_latents(latents)

            # ==========================================
            # 6. DYNAMIC DELTA MASKING (The Rigorous Fix)
            # ==========================================
            # Calculate exactly which pixels the AI changed in the high-res space
            orig_crop_np = np.array(cropped_init)
            ai_crop_np = np.array(ai_generated_crop)

            # Absolute mathematical difference between original and new AI image
            diff = cv.absdiff(orig_crop_np, ai_crop_np)
            diff_gray = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)

            # Threshold: If a pixel changed by more than 15 color values, it is part of the new object!
            _, dynamic_mask_highres = cv.threshold(diff_gray, 15, 255, cv.THRESH_BINARY)

            # Clean up the mask and slightly dilate it to fully engulf the new object's edges
            dynamic_mask_highres = cv.morphologyEx(dynamic_mask_highres, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
            dynamic_mask_highres = cv.dilate(dynamic_mask_highres, np.ones((9, 9), np.uint8), iterations=1)

            # Shrink the image AND our new dynamic mask back to the target size
            ai_generated_crop_restored = ai_generated_crop.resize((side, side), Image.LANCZOS)
            dynamic_mask_small = Image.fromarray(dynamic_mask_highres).resize((side, side), Image.NEAREST)
            dynamic_mask_small_cv = np.array(dynamic_mask_small)

            # Paste the raw square back onto the background
            full_ai_image = current_bg.copy()
            full_ai_image.paste(ai_generated_crop_restored, (x1, y1))

            # ==========================================
            # 7. POISSON BLEND: Using the Delta Mask!
            # ==========================================
            # Create a blank 512x512 mask and place our dynamic shape exactly where the crop was
            final_clone_mask = np.zeros((512, 512), dtype=np.uint8)
            final_clone_mask[y1:y1+side, x1:x1+side] = dynamic_mask_small_cv

            src = np.array(full_ai_image)
            dst = np.array(current_bg)

            # We need 3 channels for seamlessClone
            final_clone_mask_color = cv.cvtColor(final_clone_mask, cv.COLOR_GRAY2RGB)

            # Find the exact center and bounding box of the newly calculated dynamic shape
            x_d, y_d, w_d, h_d = cv.boundingRect(final_clone_mask)

            if w_d > 0 and h_d > 0:
                center = (x_d + w_d // 2, y_d + h_d // 2)
                try:
                    # Normal clone shifts colors to match the background lighting
                    blended_np = cv.seamlessClone(src, dst, final_clone_mask_color, center, cv.NORMAL_CLONE)
                    current_bg = Image.fromarray(blended_np)
                except Exception:
                    # Fallback in case seamlessClone crashes due to edge boundaries
                    current_bg = full_ai_image

        # Return the fully updated image after ALL masks have been processed
        return current_bg
