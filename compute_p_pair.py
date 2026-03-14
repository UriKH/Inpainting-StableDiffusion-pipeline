import scipy.stats as stats
import json
from utils.getters import input_output_paths_args

def compare_pipelines(json_path_a, json_path_b, metric='PickScore_raw'):
    # 1. Load the raw arrays
    with open(json_path_a, 'r') as f:
        scores_a = json.load(f)[metric]

    with open(json_path_b, 'r') as f:
        scores_b = json.load(f)[metric]

    # Sanity check: ensure both pipelines evaluated the exact same number of images
    if len(scores_a) != len(scores_b):
        print("Error: The number of evaluated images does not match!")
        return

    # 2. Run the paired t-test
    t_statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    # 3. Calculate means for context
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)

    print(f"--- Comparison Results for {metric} ---")
    print(f"Pipeline A Mean: {mean_a:.4f}")
    print(f"Pipeline B Mean: {mean_b:.4f}")
    print(f"T-Statistic:     {t_statistic:.4f}")
    print(f"P-Value:         {p_value:.4e}")
    print("-" * 40)

    # 4. Interpret the P-value (Standard alpha level is 0.05)
    if p_value < 0.05:
        if mean_b > mean_a:
            print("Conclusion: The difference IS statistically significant.")
            print("Pipeline B is definitively better.")
        else:
            print("Conclusion: The difference IS statistically significant.")
            print("Pipeline A is definitively better.")
    else:
        print("Conclusion: The difference is NOT statistically significant.")
        print("The pipelines perform similarly; the numerical gap is just variance.")


if __name__ == '__main__':
    input_paths, output_paths = input_output_paths_args()
    # Replace these with the paths to your actual metrics.json files
    compare_pipelines(input_paths, output_paths)