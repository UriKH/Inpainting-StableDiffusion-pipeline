import torch


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



