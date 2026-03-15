import torch
import gc
import math


def clear_cache() -> None:
    """
    Clear cuda cache and garbage collector to free up memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
