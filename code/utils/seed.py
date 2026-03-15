import torch
import numpy as np
import random


def set_seed(seed: int = 42):
    """
    Locks the random seed for all mathematical operations to guarantee reproducibility across runs.
    :param seed: The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
