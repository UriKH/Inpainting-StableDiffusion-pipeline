import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.getters import input_output_paths_args
from utils.seed import seed_everything
from utils.globals import MASKING_CONFIGS, COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH
from tqdm import tqdm
from PIL import Image
from coco_runner import COCODatasetGenerator

base_seed = 42


def extract_coverage_ratio(file_names, generator):
    ratios = []
    for i, img_path in tqdm(enumerate(file_names), desc="masking validation images"):
        init_image = Image.open(img_path).convert("RGB")
        img_id = generator.get_img_id(img_path)
        seed_everything(base_seed + img_id)
        _, coverage_ratio = generator.mask_generator(np.array(init_image), img_id)
        ratios.append(coverage_ratio * 100)
    return ratios


def prepare_data(validation_path, evaluation_path):
    generator = COCODatasetGenerator(COCO_INSTANCES_PATH, COCO_CAPTIONS_PATH)
    image_files_val = [os.path.join(validation_path, f) for f in os.listdir(validation_path) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files_eval = [os.path.join(evaluation_path, f) for f in os.listdir(evaluation_path) if
                        f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    df = pd.DataFrame({
        'Mask Percentage': np.concatenate([
            extract_coverage_ratio(image_files_val, generator),
            extract_coverage_ratio(image_files_eval, generator)]
        ),
        'Dataset': ['Validation'] * len(image_files_val) + ['Evaluation'] * len(image_files_eval)
    })
    return df


def draw_hist(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df, x='Mask Percentage', hue='Dataset',
        kde=True, element="step", palette=['#1f77b4', '#ff7f0e'],
        alpha=0.4
    )

    plt.xlim(0, 75)

    plt.title('Distribution of Mask Percentages per Image')
    plt.xlabel('Percentage of Image Masked (%)')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("mask_distribution_hist.png", dpi=300, bbox_inches='tight')


def draw_ecdf(df):
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=df, x='Mask Percentage', hue='Dataset', palette=['#1f77b4', '#ff7f0e'])
    plt.xlim(0, 75)
    plt.title('Cumulative Distribution of Mask Coverage')
    plt.xlabel('Mask Coverage (%)')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("mask_distribution_ecdf.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    validation_path, evaluation_path = input_output_paths_args()
    df = prepare_data(validation_path, evaluation_path)

    draw_hist(df)
    draw_ecdf(df)
