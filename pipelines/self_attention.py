import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers.models.attention_processor import AttnProcessor2_0
import math
import torch


class MaskedSelfAttnProcessor:
    def __init__(self, latent_h, latent_w):
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.mask_tensor = None  # Expected shape: (1, 1, H, W) where 1 is the hole

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        # 1. Standard projection
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 2. Dynamic Spatial Resizing for the mask
        if self.mask_tensor is not None:
            ratio = math.sqrt((self.latent_h * self.latent_w) / sequence_length)
            h_i = int(self.latent_h / ratio)
            w_i = int(self.latent_w / ratio)
            
            # Using nearest to keep the boolean logic clean for self-attention
            mask_resized = F.interpolate(self.mask_tensor, size=(h_i, w_i), mode='nearest')
            mask_flat = mask_resized.view(1, sequence_length)  # Shape: (1, N)
            
            # 3. Build the Bias Matrix B
            # Q_mask shape: (1, N, 1) -> Represents the queries
            # K_mask shape: (1, 1, N) -> Represents the keys
            Q_mask = mask_flat.unsqueeze(-1)
            K_mask = mask_flat.unsqueeze(-2)
            
            # Rule: If Query is Background (0) and Key is Hole (1), Forbid it.
            forbidden = (Q_mask == 0) & (K_mask == 1)
            
            # Create the bias tensor and fill forbidden connections with -10000
            bias = torch.zeros((1, 1, sequence_length, sequence_length), device=query.device, dtype=query.dtype)
            bias.masked_fill_(forbidden, -10000.0)
            
            # Expand to match batch size (for CFG) and heads
            bias = bias.expand(batch_size, attn.heads, sequence_length, sequence_length)
            
            # Combine with existing attention_mask if one was passed by the UNet
            if attention_mask is not None:
                attention_mask = attention_mask + bias
            else:
                attention_mask = bias

        # 4. Calculate Attention using PyTorch's optimized function
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states