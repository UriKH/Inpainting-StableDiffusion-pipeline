import torch
import gc


def clear_cache():
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
    
    # 1. Create the coordinate grid
    y = torch.arange(H, dtype=torch.float32, device=device) / H * res_h
    x = torch.arange(W, dtype=torch.float32, device=device) / W * res_w
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    
    # 2. Get integer grid coordinates and fractional coordinates
    iy, ix = gy.floor().long(), gx.floor().long()
    fy, fx = gy - iy, gx - ix
    
    # 3. The Perlin Fade Function: 6t^5 - 15t^4 + 10t^3 (ensures smooth second derivatives)
    fade_y = 6 * fy**5 - 15 * fy**4 + 10 * fy**3
    fade_x = 6 * fx**5 - 15 * fx**4 + 10 * fx**3
    
    # 4. Generate random gradients at the grid corners
    angles = torch.rand((B, C, res_h + 1, res_w + 1), device=device) * 2 * math.pi
    gx_grid = torch.cos(angles)
    gy_grid = torch.sin(angles)
    
    # 5. Calculate dot products of the gradients and distance vectors
    def dot(y_off, x_off):
        g_x = gx_grid[:, :, iy + y_off, ix + x_off]
        g_y = gy_grid[:, :, iy + y_off, ix + x_off]
        return g_x * (fx - x_off) + g_y * (fy - y_off)
        
    # 6. Interpolate along x, then y
    nx0 = torch.lerp(dot(0, 0), dot(0, 1), fade_x)
    nx1 = torch.lerp(dot(1, 0), dot(1, 1), fade_x)
    n_final = torch.lerp(nx0, nx1, fade_y)
    
    # 7. Normalize from roughly [-1, 1] to [0, 1]
    return (n_final + 1.0) / 2.0