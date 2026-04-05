import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from pipelines.cross_attention import MaskedCrossAttnProcessor
from pipelines.self_attention import MaskedSelfAttnProcessor


class Injector:
    @staticmethod
    def inject(unet, latent_h, latent_w, self_mask, cross_mask, ignore_cross_attention, ca_resize_mode, sa_resize_mode, sa_dilation_threshold):
        """
        Replaces standard cross-attention with the Masked processor.
        :param unet: The UNet model.
        :param latent_h: The height of the latent space.
        :param latent_w: The width of the latent space.
        :param self_mask: The binary mask tensor for self-attention.
        :param cross_mask: The binary mask tensor for cross-attention.
        :param ignore_cross_attention: A flag to ignore cross-attention.
        :param ca_resize_mode: The resize mode for cross-attention.
        :param sa_resize_mode: The resize mode for self-attention.
        :param sa_dilation_threshold: The dilation threshold for self-attention.
        (This function was implemented with the assistance of AI)
        """
        processor_dict = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name and self_mask is not None:  # Self-Attention Layers
                processor = MaskedSelfAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = self_mask
                processor.dilation_threshold = sa_dilation_threshold
                processor.resize_mode = sa_resize_mode
                processor_dict[name] = processor
            elif "attn2" in name and not ignore_cross_attention and cross_mask is not None:  # Cross-Attention Layers
                processor = MaskedCrossAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = cross_mask
                processor.resize_mode = ca_resize_mode
                processor_dict[name] = processor
            else:
                processor_dict[name] = AttnProcessor2_0()

        unet.set_attn_processor(processor_dict)
        return unet

    @staticmethod
    def remove(unet):
        """
        Restores the UNet to its vanilla state to prevent side effects.
        :param unet: The UNet model.
        (This function was implemented with the assistance of AI)
        """
        processor_dict = {
            name: AttnProcessor2_0() for name in unet.attn_processors.keys()
        }
        unet.set_attn_processor(processor_dict)
        return unet