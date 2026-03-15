import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from metrics import COCOInpaintingMetricsScorer


def load_data(file_paths: str) -> pd.DataFrame:
    """
    Extracts and structures the data into a pandas DataFrame.
    :param file_paths: List of paths to the JSON files containing metrics.
    """
    data = []
    for path in file_paths:
        folder_name = os.path.basename(os.path.dirname(path))
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        with open(path, 'r') as f:
            metrics_dict = json.load(f)
            for metric_name, score in metrics_dict.items():
                if isinstance(score, (int, float)):
                    data.append({
                        "Folder": folder_name,
                        "Metric": metric_name,
                        "Score": score
                    })
    return pd.DataFrame(data)

def create_heatmap(df: pd.DataFrame):
    """
    Creates a standardized heatmap across all pipelines and metrics.
    (This function was implemented by AI)
    """
    pivot_df = df.pivot(index='Folder', columns='Metric', values='Score')
    norm_df = pivot_df.copy()
    
    for metric in pivot_df.columns:
        higher_is_better = COCOInpaintingMetricsScorer.METRIC_BEST_HIGHEST.get(metric, True)
        min_val = pivot_df[metric].min()
        max_val = pivot_df[metric].max()

        if max_val == min_val:
            norm_df[metric] = 0.5 
        elif higher_is_better:
            norm_df[metric] = (pivot_df[metric] - min_val) / (max_val - min_val)
        else:
            norm_df[metric] = (max_val - pivot_df[metric]) / (max_val - min_val)

    plt.figure(figsize=(10, 10))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        norm_df, 
        annot=pivot_df,          # Show true scores
        fmt=".3g",               # Format to 3 significant digits
        cmap="flare",            # Dark blue = best performance
        cbar=False,              # Hide colorbar since scales are mixed
        linewidths=.5,
        linecolor='lightgray'
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("metrics_heatmap.png", dpi=300)
    print("Saved metrics_heatmap.png")

def create_radar_chart(df: pd.DataFrame, selected_folders=None):
    """
    Creates a radar chart for comparing specific pipelines.
    (This function was implemented by AI)
    """
    if selected_folders:
        df = df[df['Folder'].isin(selected_folders)]

    pivot_df = df.pivot(index='Folder', columns='Metric', values='Score')
    metrics = list(pivot_df.columns)
    num_vars = len(metrics)

    norm_df = pivot_df.copy()
    axis_labels = []
    bounds_info = []

    max_metric_len = max(len(m) for m in metrics)
    
    for metric in metrics:
        higher_is_better = COCOInpaintingMetricsScorer.METRIC_BEST_HIGHEST.get(metric, True)
        min_val = pivot_df[metric].min()
        max_val = pivot_df[metric].max()

        if max_val == min_val:
            norm_df[metric] = 1.0
            worst_val, best_val = min_val, max_val
        elif higher_is_better:
            norm_df[metric] = (pivot_df[metric] - min_val) / (max_val - min_val)
            worst_val, best_val = min_val, max_val
        else:
            norm_df[metric] = (max_val - pivot_df[metric]) / (max_val - min_val)
            worst_val, best_val = max_val, min_val

        axis_labels.append(metric)
        bounds_info.append(f"{metric:<{max_metric_len}} {worst_val:<6.3g} to {best_val:<6.3g}")

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(18, 7), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], axis_labels, size=11, fontweight='medium')

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle == 0:
            label.set_horizontalalignment('left')
            label.set_verticalalignment('center')
        elif 0 < angle < pi / 2:
            label.set_horizontalalignment('left')
            label.set_verticalalignment('bottom')
        elif angle == pi / 2:
            label.set_horizontalalignment('center')
            label.set_verticalalignment('bottom')
        elif pi / 2 < angle < pi:
            label.set_horizontalalignment('right')
            label.set_verticalalignment('bottom')
        elif angle == pi:
            label.set_horizontalalignment('right')
            label.set_verticalalignment('center')
        elif pi < angle < 3 * pi / 2:
            label.set_horizontalalignment('right')
            label.set_verticalalignment('top')
        elif angle == 3 * pi / 2:
            label.set_horizontalalignment('center')
            label.set_verticalalignment('top')
        else:
            label.set_horizontalalignment('left')
            label.set_verticalalignment('top')

    ax.tick_params(axis='x', pad=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)
    colors = sns.color_palette("husl", len(norm_df.index))

    for idx, (folder_name, row) in enumerate(norm_df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=folder_name, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.2)

    plt.legend(title="Pipelines", loc='lower left', bbox_to_anchor=(1.25, 0.52), fontsize=11)
    bounds_text = "Metric Bounds\n" + "-" * 32 + "\n" + "\n\n".join(bounds_info)
    ax.text(1.25, 0.48, bounds_text, transform=ax.transAxes, fontsize=10,
            family='monospace',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="lightgray", alpha=0.9))

    plt.subplots_adjust(left=0.05, right=0.50)
    plt.savefig("experiment_radar_chart.png", dpi=300, bbox_inches='tight')
    print("Saved experiment_radar_chart.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Heatmap and Radar charts from JSON metric files.")
    parser.add_argument('-j', '--json-files', help='Paths to the global directory containing metrics JSON files.')
    parser.add_argument('-c', '--compare', nargs='+', help='Specific folder names to compare in the radar chart (e.g., -c v1 v3 v4)')
    
    args = parser.parse_args()
    
    filenames = os.listdir(args.json_files)
    file_paths = [os.path.join(args.json_files, filename, "metrics.json") for filename in filenames]
    
    df = load_data(file_paths)
    
    if not df.empty:
        create_heatmap(df)
        create_radar_chart(df, selected_folders=args.compare)
    else:
        print("No valid data found to plot.")
