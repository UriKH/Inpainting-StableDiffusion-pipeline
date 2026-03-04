# import argparse
# import json
#
# from metrics import COCOInpaintingMetricsScorer
# import numpy as np
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("metrics_path", help="Path to the metrics JSON file")
#
#     args = parser.parse_args()
#
#     with open(args.metrics_path, 'r') as f:
#         metrics = json.load(f)
#
#     for method in COCOInpaintingMetricsScorer.METRICS:
#         results = np.array([metric[method] for metric in metrics.values()])
#         print(f'{"=" * 10} {method} {"=" * 10}')
#         print(f"Average: {np.mean(results):.3f}")
#         print(f"Median: {np.median(results):.3f}")
#         print(f"Standard deviation: {np.std(results):.3f}")