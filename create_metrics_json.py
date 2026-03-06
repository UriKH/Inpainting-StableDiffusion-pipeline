from metrics import COCOInpaintingMetricsScorer
from utils.getters import input_output_paths_args
import os
import json
from tqdm import tqdm
from pprint import pprint


if __name__ == "__main__":
    input_paths, output_paths = input_output_paths_args()
    scorer = COCOInpaintingMetricsScorer()
    filenames = os.listdir(input_paths)
    # scores = {
    #     filename: scorer.score(str(os.path.join(input_paths, filename)), str(os.path.join(output_paths, filename)))
    #     for filename in tqdm(filenames, desc="Calculating metrics...")
    # }

    for filename in tqdm(filenames, desc="Calculating metrics..."):
        scorer.update_metrics(str(os.path.join(input_paths, filename)), str(os.path.join(output_paths, filename)))

    scores = scorer.compute_metrics()
    pprint(scores)

    metrics_path = os.path.join(output_paths, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(scores, f)
    print(f'Computed metrics dumped to: {metrics_path}')

