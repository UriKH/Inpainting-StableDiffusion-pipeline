import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import ToTensor
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
fixed_versions_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(fixed_versions_dir)

sys.path.append(fixed_versions_dir)
sys.path.append(root_dir)

from fixed_versions.generators.coco_runner import COCODatasetGenerator
from utils.globals import COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH


class COCOInpaintingMetricsScorer:
    FID = 'FID'
    SSIM = 'SSIM'
    LPIPS = 'LPIPS'
    CLIP_SCORE = 'CLIP score'
    METRICS = [FID, SSIM, LPIPS, CLIP_SCORE]

    def __init__(self, device="cuda"):
        self.device = device

        # Distribution / Realism Metric
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

        # Text-Alignment Metric
        print("Loading native Hugging Face CLIP model for robust scoring...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_scores = [] 

        # Reconstruction Metrics
        # data_range=1.0 because ToTensor() scales images to [0.0, 1.0]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        # normalize=True tells LPIPS to expect [0.0, 1.0] inputs instead of [-1.0, 1.0]
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)
        self.coco_manager = COCODatasetGenerator(COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH)

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts PIL Image to the format expected by torchmetrics (1, C, H, W) float tensor [0.0, 1.0]"""
        tensor = ToTensor()(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    def update_fid_clip(self, real_image: Image.Image, generated_image: Image.Image, prompt: str):
        """Updates the FID and CLIP Score states."""
        real_tensor = self.preprocess_image(real_image)
        gen_tensor = self.preprocess_image(generated_image)

        self.fid.update(real_tensor, real=True)
        self.fid.update(gen_tensor, real=False)
        
        # 2. Calculate mathematical CLIP Score natively
        inputs = self.clip_processor(text=[prompt], images=generated_image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)

            # Extract raw embeddings
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize to unit vectors
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)

            # Calculate cosine similarity and scale by 100
            cos_sim = torch.sum(image_embeds * text_embeds, dim=-1)
            score = 100 * torch.max(cos_sim, torch.zeros_like(cos_sim))

            self.clip_scores.append(score.item())

    def update_reconstruction(self, real_image: Image.Image, generated_image: Image.Image):
        """Updates the SSIM and LPIPS states with a 1-to-1 image comparison."""
        real_tensor = self.preprocess_image(real_image)
        gen_tensor = self.preprocess_image(generated_image)

        self.ssim.update(gen_tensor, real_tensor)
        self.lpips.update(gen_tensor, real_tensor)

    def compute_metrics(self) -> dict:
        """Calculates and returns the final scores across all updated images."""
        print("Computing final metrics... this might take a moment.")
        
        mean_clip = sum(self.clip_scores) / len(self.clip_scores) if self.clip_scores else 0.0

        results = {
            self.FID: float(self.fid.compute()),
            self.CLIP_SCORE: float(mean_clip),
            self.SSIM: float(self.ssim.compute()),
            self.LPIPS: float(self.lpips.compute())
        }

        # Reset states for the next run
        self.fid.reset()
        self.ssim.reset()
        self.lpips.reset()
        return results

    def update_metrics(self, real_image_path: str, generated_image_path: str):
        def enforce_min_bbox(bbox: tuple, min_size=32, img_width=512, img_height=512) -> tuple:
            """Ensures a bounding box is at least min_size x min_size to prevent CNN pooling crashes."""
            left, top, w, h = bbox
            right = left + w
            bottom = top + h

            if w < min_size:
                pad_w = (min_size - w) // 2
                left = max(0, left - pad_w)
                right = min(img_width, left + min_size)
            if h < min_size:
                pad_h = (min_size - h) // 2
                top = max(0, top - pad_h)
                bottom = min(img_height, top + min_size)
            return int(left), int(top), int(right), int(bottom)
        
        try:
            prompt, bbox = self.coco_manager.get_mask_prompt(real_image_path)
        except Exception as e:
            print(f'unexpected excpetion: {e} (continue anyway!)')
            return
        real_image = Image.open(real_image_path).convert("RGB")
        generated_image = Image.open(generated_image_path).convert("RGB")
        
        width, height = real_image.size
        bbox = enforce_min_bbox(bbox, img_width=width, img_height=height)
        cropped_real = real_image.crop(bbox)
        cropped_gen = generated_image.crop(bbox)
        self.update_fid_clip(real_image, generated_image, prompt)
        self.update_reconstruction(cropped_real, cropped_gen)
