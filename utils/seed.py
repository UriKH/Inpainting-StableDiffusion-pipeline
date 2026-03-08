import torch
import numpy as np
import random


def seed_everything(seed: int = 42):
    """
    Locks the random seed for all mathematical operations to guarantee
    100% reproducible results across runs.
    """
    # 1. Standard Python Random
    random.seed(seed)

    # 2. NumPy Random (Crucial for any np.random calls)
    np.random.seed(seed)

    # 3. PyTorch CPU
    torch.manual_seed(seed)

    # 4. PyTorch GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Force CuDNN to be strictly deterministic (slightly slower, but mathematically exact)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
