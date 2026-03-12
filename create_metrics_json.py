from metrics import COCOInpaintingMetricsScorer
from utils.getters import input_output_paths_args
import os
import json
from tqdm import tqdm
from pprint import pprint


def compute_metrics(input_paths: str, output_paths: str):
    """
    Computes metrics for the images in the input directory.
    :param input_paths: Path to the input directory containing images.
    :param output_paths: Path to the output directory where metrics.json will be saved.
    """
    scorer = COCOInpaintingMetricsScorer()
    filenames = os.listdir(input_paths)

    for filename in tqdm(filenames, desc="Calculating metrics..."):
        scorer.update_metrics(str(os.path.join(input_paths, filename)), str(os.path.join(output_paths, filename)))

    scores = scorer.compute_metrics()
    pprint(scores)

    metrics_path = os.path.join(output_paths, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(scores, f)
    print(f'Computed metrics dumped to: {metrics_path}')


if __name__ == "__main__":
    input_paths, output_paths = input_output_paths_args()
    compute_metrics(input_paths, output_paths)

