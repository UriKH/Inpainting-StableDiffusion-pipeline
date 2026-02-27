from vanilla_pipeline import InpaintPipeline


class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self):
        super().__init__()

    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        