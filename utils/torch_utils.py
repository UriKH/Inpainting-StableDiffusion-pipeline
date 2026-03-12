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


def generate_perlin_noise_2d(shape, res, device):
    """
    Generates true Gradient Noise (Perlin) for organic blob structures.
    shape: (B, C, H, W)
    res: (res_h, res_w) - the number of Perlin grid cells (defines blob size)
    """
    B, C, H, W = shape
    res_h, res_w = res

    y = torch.arange(H, dtype=torch.float32, device=device) / H * res_h
    x = torch.arange(W, dtype=torch.float32, device=device) / W * res_w
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    iy, ix = gy.floor().long(), gx.floor().long()
    fy, fx = gy - iy, gx - ix
    
    fade_y = 6 * fy**5 - 15 * fy**4 + 10 * fy**3
    fade_x = 6 * fx**5 - 15 * fx**4 + 10 * fx**3
    angles = torch.rand((B, C, res_h + 1, res_w + 1), device=device) * 2 * math.pi
    gx_grid = torch.cos(angles)
    gy_grid = torch.sin(angles)

    def dot(y_off, x_off):
        g_x = gx_grid[:, :, iy + y_off, ix + x_off]
        g_y = gy_grid[:, :, iy + y_off, ix + x_off]
        return g_x * (fx - x_off) + g_y * (fy - y_off)
        
    nx0 = torch.lerp(dot(0, 0), dot(0, 1), fade_x)
    nx1 = torch.lerp(dot(1, 0), dot(1, 1), fade_x)
    n_final = torch.lerp(nx0, nx1, fade_y)
    return (n_final + 1.0) / 2.0