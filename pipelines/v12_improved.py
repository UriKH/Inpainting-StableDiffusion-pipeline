from pipelines.v10_improved import ImprovedInpaintPipelineV10
import torch


class ImprovedInpaintPipelineV12(ImprovedInpaintPipelineV10):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

