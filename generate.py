from vanilla_pipeline import InpaintPipeline as VanillaPipeLine
from ready_pipeline import PipeLine as ReadyPipeLine
import torch 


if __name__ == "__main__":
    #p = VanillaPipeLine()
    #p.apply_multiple(r'./data/ours/masked', r'./data/ours/results_vanilla')
    
    #del p    
    #torch.cuda.empty_cache()

    p = ReadyPipeLine()
    p.apply_multiple(r'./data/ours/masked', r'./data/ours/results_ready_made')

