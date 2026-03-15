import torch
from diffusers.models.attention_processor import AttnProcessor2_0

class Injector:
    @staticmethod
    def inject(unet, latent_h, latent_w, mask_tensor, ignore_cross_attention, ca_resize_mode):
        """
        Replaces standard cross-attention with the Masked processor.
        (This function was implemented with the assistance of AI)
        """
        processor_dict = {}
        for name in unet.attn_processors.keys():
            if "attn2" in name and not ignore_cross_attention:
                processor = MaskedCrossAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = mask_tensor
                processor.resize_mode = ca_resize_mode
                processor_dict[name] = processor
            else:
                processor_dict[name] = AttnProcessor2_0()
        unet.set_attn_processor(processor_dict)
        return unet

    @staticmethod
    def remove():
        raise NotImplementedError