import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import ToTensor
from PIL import Image

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
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(self.device)

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
        self.clip_score.update(gen_tensor, [prompt])

    def update_reconstruction(self, real_image: Image.Image, generated_image: Image.Image):
        """Updates the SSIM and LPIPS states with a 1-to-1 image comparison."""
        real_tensor = self.preprocess_image(real_image)
        gen_tensor = self.preprocess_image(generated_image)

        self.ssim.update(gen_tensor, real_tensor)
        self.lpips.update(gen_tensor, real_tensor)

    def compute_metrics(self) -> dict:
        """Calculates and returns the final scores across all updated images."""
        print("Computing final metrics... this might take a moment.")

        results = {
            self.FID: float(self.fid.compute()),
            self.CLIP_SCORE: float(self.clip_score.compute()),
            self.SSIM: float(self.ssim.compute()),
            self.LPIPS: float(self.lpips.compute())
        }

        # Reset states for the next run
        self.fid.reset()
        self.clip_score.reset()
        self.ssim.reset()
        self.lpips.reset()
        return results

    def score(self, real_image_path: str, generated_image_path: str):
        prompt, bbox = self.coco_manager.get_mask_prompt(real_image_path)
        real_image = Image.open(real_image_path).convert("RGB")
        generated_image = Image.open(generated_image_path).convert("RGB")

        cropped_real = real_image.crop(bbox)
        cropped_gen = generated_image.crop(bbox)
        self.update_fid_clip(real_image, generated_image, prompt)
        self.update_reconstruction(cropped_real, cropped_gen)
        return self.compute_metrics()
