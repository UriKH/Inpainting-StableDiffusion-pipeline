from pipelines.v8_improved import ImprovedInpaintPipelineV8
from pipelines.pipeline import InpaintPipelineInput


class ImprovedInpaintPipelineV9(ImprovedInpaintPipelineV8):
    def __init__(self, jump_length=10, jump_n_sample=2):
        super().__init__()
        # Store resampling hyperparameters as instance variables 
        # to preserve the base pipe() signature.
        self.jump_length = jump_length
        self.jump_n_sample = jump_n_sample
    
    def apply_freeu(self, s1=0.9, s2=0.2, b1=1.2, b2=1.4):
        """
        Modifies the UNet activations during the forward pass to prioritize 
        semantic structure over high-frequency background matching.
        """
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def remove_freeu(self):
        """Restores standard UNet architecture scaling."""
        self.unet.disable_freeu()
    
    def pipe(self, pipe_in: InpaintPipelineInput):
        self.apply_freeu(s1=1, s2=1, b1=1.05, b2=1.05)
        res = super().pipe(pipe_in)
        self.remove_freeu()
        return res
