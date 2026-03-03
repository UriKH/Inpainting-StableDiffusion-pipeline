import torch


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_title(msg, n=100):
    print()
    print(n * "=")
    t = (n - len(msg)) // 2
    delta = n - 2 * t - len(msg)
    print(((t-1) * "=") + f' {msg} ' + ((t-1 + delta) * "="))
    print(n * "=")

