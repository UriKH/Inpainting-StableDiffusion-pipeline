import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import ToTensor
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import sys
import os
import numpy as np
from DISTS_pytorch import DISTS

current_dir = os.path.dirname(os.path.abspath(__file__))
fixed_versions_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(fixed_versions_dir)
sys.path.append(fixed_versions_dir)
sys.path.append(root_dir)

from coco_runner import COCODatasetGenerator
from mask_generator import MaskGenerator
from utils.globals import COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH, MASKING_CONFIGS
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel

class COCOInpaintingMetricsScorer:
    """
    COCO Inpainting Metrics Scorer.
    """

    FID = 'FID'
    SSIM = 'SSIM'
    LPIPS = 'LPIPS'
    CLIP_SCORE = 'CLIP score'
    MSE = 'MSE'
    PSNR = 'PSNR'
    DISTS = 'DISTS'
    DINO_VITS = 'DINOv2'
    PICK_SCORE = 'PickScore'
    METRICS = [FID, SSIM, LPIPS, CLIP_SCORE, PICK_SCORE, MSE, PSNR, DISTS, DINO_VITS]
    METRIC_BEST_HIGHEST = {
        FID: False, SSIM: True, PICK_SCORE: True, LPIPS: False, CLIP_SCORE: True, MSE: False, PSNR: True, DISTS: False, DINO_VITS: True
    }

    def __init__(self, device="cuda"):
        self.device = device

        # Distribution / Realism Metric
        print(f'Loading FID model...')
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

        # Text-Alignment Metric
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
        self.clip_scores = []

        print("Loading PickScore model...")
        self.pick_processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
        self.pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(self.device)
        self.pick_scores = []

        # Reconstruction Metrics
        print('Loading SSIM model...')
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        # normalize=True tells LPIPS to expect [0.0, 1.0] inputs instead of [-1.0, 1.0]
        print('Loading LPIPS model...')
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)
        self.coco_manager = COCODatasetGenerator(COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH)

        print('Loading SSIM model...')
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        print('Loading MSE and PSNR models...')
        self.mse = MeanSquaredError().to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        print('Loading DISTS model...')
        self.dists = DISTS().to(self.device)
        self.dists_scores = []

        print('Loading DINOv2 model...')
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dinov2.eval()

        self.dinov2_scores = []
        self.ssim_scores = []
        self.lpips_scores = []
        self.mse_scores = []
        self.psnr_scores = []

        self.ratio_buckets = {f'{i}-{i+5}': 0 for i in range(0, 91, 5)}
        self.bucket_keys_map = {i: f'{i}-{i+5}' for i in range(0, 91, 5)}

        self.mask_generator = MaskGenerator(**MASKING_CONFIGS)

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Converts PIL Image to the format expected by torchmetrics - a float tensor.
        :param pil_image: PIL Image to preprocess
        """
        tensor = ToTensor()(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    def update_fid_clip_dists_dino(self, real_image: Image.Image, generated_image: Image.Image, prompt: str):
        """
        Updates the FID, CLIP Score, DISTS and DINOv2 states.
        :param real_image: Real image to compare against
        :param generated_image: Generated image to compare
        :param prompt: Text prompt to use for CLIP Score
        """
        real_tensor = self.preprocess_image(real_image)
        gen_tensor = self.preprocess_image(generated_image)
        self.fid.update(real_tensor, real=True)
        self.fid.update(gen_tensor, real=False)
        self.dists_scores.append(self.dists(real_tensor, gen_tensor))
        self.dinov2_scores.append(self._get_dino_similarity(real_tensor, gen_tensor))

        # Compute CLIP Score manually
        inputs = self.clip_processor(text=[prompt], images=generated_image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            cos_sim = F.cosine_similarity(image_embeds, text_embeds, dim=-1)
            score = 100 * torch.clamp(cos_sim, min=0.0)
            self.clip_scores.append(score.item())

        pick_inputs = self.pick_processor(
            images=generated_image,
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # Pass everything through the model at once using ** unpacking
            pick_outputs = self.pick_model(**pick_inputs)

            # Extract the raw tensors from the HuggingFace output object
            image_embs = pick_outputs.image_embeds
            text_embs = pick_outputs.text_embeds

            # Now we can safely normalize the tensors
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # PickScore is the scaled dot product
            pick_score = self.pick_model.logit_scale.exp() * (text_embs @ image_embs.T)[0, 0]
            self.pick_scores.append(pick_score.item())


    def update_reconstruction(self, real_image: Image.Image, generated_image: Image.Image):
        """
        Updates the SSIM and LPIPS states with.
        :param real_image: Real image to compare against
        :param generated_image: Generated image to compare
        """
        real_tensor = self.preprocess_image(real_image)
        gen_tensor = self.preprocess_image(generated_image)

        self.ssim_scores.append(float(self.ssim(gen_tensor, real_tensor)))
        self.lpips_scores.append(float(self.lpips(gen_tensor, real_tensor)))
        self.mse_scores.append(float(self.mse(gen_tensor, real_tensor)))
        self.psnr_scores.append(float(self.psnr(gen_tensor, real_tensor)))

        # self.ssim.update(gen_tensor, real_tensor)
        # self.lpips.update(gen_tensor, real_tensor)
        # self.mse.update(gen_tensor, real_tensor)
        # self.psnr.update(gen_tensor, real_tensor)

    def compute_metrics(self) -> dict:
        """
        Calculates and returns the final scores across all updated images.
        :return: Dictionary containing the computed metrics in the format {metric_name: metric_value}
        """
        print("Computing final metrics... this might take a moment.")

        mean_clip = sum(self.clip_scores) / len(self.clip_scores) if self.clip_scores else 0.0
        mean_pick = sum(self.pick_scores) / len(self.pick_scores) if self.pick_scores else 0.0
        results = {
            self.FID: float(self.fid.compute()),
            self.CLIP_SCORE: float(mean_clip),
            self.SSIM: float(self.ssim.compute()),
            self.LPIPS: float(self.lpips.compute()),
            self.MSE: float(self.mse.compute()),
            self.PSNR: float(self.psnr.compute()),
            self.PICK_SCORE: float(mean_pick),
            self.DISTS: float(sum(self.dists_scores) / len(self.dists_scores)),
            self.DINO_VITS: float(sum(self.dinov2_scores) / len(self.dinov2_scores)),
            'ratio buckets': self.ratio_buckets
        }

        # Reset states for the next run
        self.fid.reset()
        self.ssim.reset()
        self.lpips.reset()
        self.mse.reset()
        self.psnr.reset()
        raws = {
            self.CLIP_SCORE: [float(score) for score in self.clip_scores],
            self.PICK_SCORE: [float(score) for score in self.pick_scores],
            self.DISTS: [float(score) for score in self.dists_scores],
            self.DINO_VITS: [float(score) for score in self.dinov2_scores],
            self.SSIM: self.ssim_scores,
            self.MSE: self.mse_scores,
            self.PSNR: self.psnr_scores,
            self.LPIPS: self.lpips_scores,
        }
        self.ssim_scores = []
        self.clip_scores = []
        self.pick_scores = []
        self.dists_scores = []
        self.dinov2_scores = []
        self.lpips_scores = []
        self.mse_scores = []
        self.psnr_scores = []
        return results, raws

    def update_metrics(self, real_image_path: str, generated_image_path: str):
        """
        Updates the metrics with the given real and generated images.
        :param real_image_path: Path to the real image
        :param generated_image_path: Path to the generated image
        """
        try:
            prompt, img_id = self.coco_manager.get_prompt_img_id(real_image_path)
        except Exception as e:
            print(f'unexpected exception: {e} (continue anyway!)')
            return
        try:
            real_image = Image.open(real_image_path).convert("RGB")
            generated_image = Image.open(generated_image_path).convert("RGB")
        except FileNotFoundError as e:
            print(f'no generated image in {generated_image_path} ...: {e}')
            return

        _, coverage = self.mask_generator(np.array(real_image), img_id)
        for i in range(0, 91, 5):
            if i <= coverage * 100 < i+5:
                self.ratio_buckets[self.bucket_keys_map[i]] += 1
                break

        self.update_fid_clip_dists_dino(real_image, generated_image, prompt)
        self.update_reconstruction(real_image, generated_image)

    def _get_dino_similarity(self, img1, img2):
        """
        Compute DINOv2 similarity between two images.
        :param img1: First image tensor
        :param img2: Second image tensor
        """
        with torch.no_grad():
            feat1 = self.dinov2(self._prepare_for_dino(img1))
            feat2 = self.dinov2(self._prepare_for_dino(img2))
            sim = F.cosine_similarity(feat1, feat2)
        return sim.mean()

    @staticmethod
    def _prepare_for_dino(img_tensor, patch_size: int = 14):
        """
        Standardizes image size for DINOv2 without distortion using the 'training' normalization parameters.
        :param img_tensor: Input image tensor
        :param patch_size: Size of the patch to extract features from (default: 14)

        (This function was implemented with the use of AI)
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)
        img_normalized = (img_tensor - mean) / std

        h, w = img_normalized.shape[-2:]
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        img_padded = F.pad(img_normalized, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return img_padded