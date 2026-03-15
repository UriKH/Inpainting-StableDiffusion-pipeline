import torch
import numpy as np
from PIL import Image


class VaeConverter:
    @staticmethod
    def pil_to_tensor(image, device):
        image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
        return image_tensor

    @staticmethod
    def tensor_to_pil(tensor):
        image = (tensor / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)
