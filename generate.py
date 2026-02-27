from vanilla_pipeline import InpaintPipeline
import torch


if __name__ == "__main__":
    p = InpaintPipeline()
    p.apply_multiple(r'./data/ours/masked', r'./data/ours/results2')


