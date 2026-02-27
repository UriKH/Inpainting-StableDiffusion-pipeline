from vpipeline import PipeLine
import torch


MODEL_ID = "Manojb/stable-diffusion-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    p = PipeLine(MODEL_ID, DEVICE)
    p.apply_multiple(r'./data/ours/masked', r'./data/ours/results')


    