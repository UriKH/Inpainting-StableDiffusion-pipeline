from pipelines.v10_improved import ImprovedInpaintPipelineV10
from pipelines.pipeline import InpaintPipelineInput


class ImprovedInpaintPipelineV11(ImprovedInpaintPipelineV10):
    def __init__(self, freeu_s1=0.95, freeu_s2=0.8, freeu_b1=1.1, freeu_b2=1.1, use_freeu=True, **kwargs):
        super().__init__(**kwargs)
        self.freeu_s1 = freeu_s1
        self.freeu_s2 = freeu_s2
        self.freeu_b1 = freeu_b1
        self.freeu_b2 = freeu_b2
        self.use_freeu = use_freeu
    
    def apply_freeu(self):
        """
        Modifies the UNet activations during the forward pass to prioritize 
        semantic structure over high-frequency background matching.
        """
        self.unet.enable_freeu(s1=self.freeu_s1, s2=self.freeu_s2, b1=self.freeu_b1, b2=self.freeu_b2)

    def remove_freeu(self):
        """Restores standard UNet architecture scaling."""
        self.unet.disable_freeu()
    
    def pipe(self, pipe_in: InpaintPipelineInput):
        if self.use_freeu:
            self.apply_freeu()
        res = super().pipe(pipe_in)
        if self.use_freeu:
            self.remove_freeu()
        return res
