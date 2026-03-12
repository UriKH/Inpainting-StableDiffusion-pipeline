import importlib
import argparse
import traceback
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coco_runner import COCODatasetGenerator
from utils.getters import input_output_paths_args
from utils.globals import COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH
from utils.torch_utils import clear_cache


def generate(input_paths, output_paths, pipeline):
    clear_cache()
    generator = COCODatasetGenerator(
        COCO_INSTANCES_PATH,
        COCO_CAPTIONS_PATH,
    )
    generator.generate(input_paths, output_paths, pipeline)


def main(args):
    i = int(args.version)
    if i == 0:
        module_name = 'pipelines.vanilla_pipeline'
        class_name = 'InpaintPipelineVanilla'
    else:
        module_name = f'pipelines.v{i}_improved' 
        class_name = f'ImprovedInpaintPipelineV{i}'
        
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            print(f"Error: Module '{module_name}' not found.")
        else:
            print(f"Error while importing '{module_name}': missing dependency '{e.name}'.")
            traceback.print_exc()
        return
    except ImportError as e:
        print(f"ImportError while loading '{module_name}': {e}")
        traceback.print_exc()
        return

    try:
        target_class = getattr(module, class_name)
        params = vars(args)
        params.pop('version')
        instance = target_class(**params)
    except AttributeError:
        print(f"Error: Class '{class_name}' not found in '{module_name}'.")
        return

    try:
        generate(input_paths, output_paths, instance)
    except Exception as e:
        print(f"Error while running generator: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="which pipeline version to import")

    # Self-attention dilation thresholds
    parser.add_argument("--sa_dilation_threshold", default=0.0, type=float, help="binary interpolation threshold for self attention")
    parser.add_argument("--sa_resize_mode", default='nearest', type=str, help="interpolation mode for self attention")
    parser.add_argument("--ca_resize_mode", default='nearest', type=str, help="interpolation mode for cross attention")
    parser.add_argument("--ignore_cross_attention", action='store_true', help="ignore cross attention upgrades")

    # Preprocessing mask dilation and feathering
    parser.add_argument("--pp_feather_radius", default=5, type=int, help="feathering kernel size for preprocessing")
    parser.add_argument("--pp_dilate_kernel_size", default=3, type=int, help="dilation kernel size for preprocessing")
    parser.add_argument("--use_negative_prompt", action='store_true', help="dilation kernel size for preprocessing")
    parser.add_argument("--ignore_improvement_v5", action='store_true', help="remove effect of V5 (mask preprocessing and negative prompt)")

    # Soft masking in cross-attention
    parser.add_argument("--sm_dilation_kernel", default=3, type=int, help="dilation kernel for soft masking")
    parser.add_argument("--sm_blur_kernel", default=5, type=int, help="blurring kernel for soft masking")
    parser.add_argument("--sm_sigma", default=5.0, type=float, help="sigma for soft masking")
    parser.add_argument("--use_sm_in_sa", action='store_true', help="use soft masking in self-attention")

    # resampling time travel
    parser.add_argument("--rp_jump_length", default=7, type=int, help="jump length for resampling")
    parser.add_argument("--rp_jump_n_sample", default=3, type=int, help="number of jumps for resampling")
    parser.add_argument("--ds_min_jumps", default=1, type=int, help="min number of jumps for resampling")
    parser.add_argument("--ds_max_jumps", default=4, type=int, help="max number of jumps for resampling")
    parser.add_argument("--ds_min_jump_len", default=5, type=int, help="min jump length for resampling")
    parser.add_argument("--ds_max_jump_len", default=10, type=int, help="max jump length for resampling")
    parser.add_argument("--use_dynamic_schedule", action='store_true', help="use dynamic schedule for resampling")

    # Dynamic Mask Blending
    parser.add_argument("--dmb_dilation_kernel_size", default=3, type=int,
                        help="dynamic mask blending dilation kernel size")
    parser.add_argument("--dmb_blur_kernel_size", default=5, type=int, help="dynamic mask blending blur kernel size")
    parser.add_argument("--dmb_sigma", default=5.0, type=float, help="dynamic mask blending sigma")

    # Organic masking using Perlin noise
    parser.add_argument("--om_noise_res", default=4, type=int, help="noise resolution for organic masking")
    parser.add_argument("--om_dilation_kernel", default=7, type=int, help="dilation kernel for organic masking")
    parser.add_argument("--om_thresh", default='linear', type=str, help="dilation kernel for organic masking")

    parser.add_argument("--freeu_s1", default=0.95, type=float, help="s1 parameter for FreeU")
    parser.add_argument("--freeu_s2", default=0.8, type=float, help="s2 parameter for FreeU")
    parser.add_argument("--freeu_b1", default=1.1, type=float, help="b1 parameter for FreeU")
    parser.add_argument("--freeu_b2", default=1.1, type=float, help="b2 parameter for FreeU")
    parser.add_argument("--use_freeu", action='store_true', help="use FreeU")

    parser.add_argument("--version", type=int, required=True, help="insert the version number (0 for vanilla)")
    parser.add_argument("--init_noise_strength", default=1.0, type=float, help="init noise strength")
    parser.add_argument("--reconstruction", action='store_true', help="reconstruction or replace")

    input_paths, output_paths = input_output_paths_args(parser)
    args = parser.parse_args()
    main(args)
