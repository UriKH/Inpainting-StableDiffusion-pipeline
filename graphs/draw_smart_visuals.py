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

def load_data(file_paths):
    """Extracts and structures the data into a pandas DataFrame."""
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

def create_heatmap(df):
    """Creates a standardized heatmap across all pipelines and metrics."""
    # Pivot the dataframe so rows are Pipelines (Folders) and columns are Metrics
    pivot_df = df.pivot(index='Folder', columns='Metric', values='Score')
    
    # We need to normalize the data to a 0-1 scale purely for the COLOR map, 
    # so that metrics with different scales don't distort the visual.
    # 1.0 will always represent the 'best' score for that specific metric.
    norm_df = pivot_df.copy()
    
    for metric in pivot_df.columns:
        higher_is_better = COCOInpaintingMetricsScorer.METRIC_BEST_HIGHEST.get(metric, True)
        min_val = pivot_df[metric].min()
        max_val = pivot_df[metric].max()
        
        # Avoid division by zero if all scores are identical
        if max_val == min_val:
            norm_df[metric] = 0.5 
        elif higher_is_better:
            norm_df[metric] = (pivot_df[metric] - min_val) / (max_val - min_val)
        else:
            # Invert lower-is-better metrics so that the lowest value gets 1.0 (dark blue)
            norm_df[metric] = (max_val - pivot_df[metric]) / (max_val - min_val)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="white")
    
    # Create the heatmap. We use norm_df for the colors, but pivot_df for the text annotations!
    ax = sns.heatmap(
        norm_df, 
        annot=pivot_df,          # Show true scores
        fmt=".3g",               # Format to 3 significant digits
        cmap="Blues",            # Dark blue = best performance
        cbar=False,              # Hide colorbar since scales are mixed
        linewidths=.5,
        linecolor='lightgray'
    )
    
    plt.title("Pipeline Comparison Heatmap (Darker = Better)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Pipeline Version", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("metrics_heatmap.png", dpi=300)
    print("Saved metrics_heatmap.png")

def create_radar_chart(df, selected_folders=None):
    """Creates a radar chart for comparing specific pipelines."""
    # If no specific folders are provided, use all of them (can get messy if > 4)
    if selected_folders:
        df = df[df['Folder'].isin(selected_folders)]
        
    pivot_df = df.pivot(index='Folder', columns='Metric', values='Score')
    metrics = list(pivot_df.columns)
    num_vars = len(metrics)
    
    # Normalize data so the outer edge of the radar is ALWAYS the best score
    norm_df = pivot_df.copy()
    for metric in metrics:
        higher_is_better = COCOInpaintingMetricsScorer.METRIC_BEST_HIGHEST.get(metric, True)
        min_val = pivot_df[metric].min()
        max_val = pivot_df[metric].max()
        
        if max_val == min_val:
            norm_df[metric] = 1.0 
        elif higher_is_better:
            norm_df[metric] = (pivot_df[metric] - min_val) / (max_val - min_val)
        else:
            norm_df[metric] = (max_val - pivot_df[metric]) / (max_val - min_val)

    # Compute angle of each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] # Close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], metrics, size=11)
    
    # Remove radial labels (0.2, 0.4, etc.) to keep it clean
    ax.set_yticklabels([])
    
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, len(norm_df.index)))

    # Plot each pipeline
    for idx, (folder_name, row) in enumerate(norm_df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1] # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=folder_name, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.1)

    plt.title("Radar Chart: Normalized Metric Performance\n(Further Out = Better)", size=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("experiment_radar_chart.png", dpi=300)
    print("Saved experiment_radar_chart.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Heatmap and Radar charts from JSON metric files.")
    parser.add_argument('-j', '--json-files', help='Paths to the global directory containing metrics JSON files.')
    # Optional: pass specific folders to compare in the radar chart
    parser.add_argument('-c', '--compare', nargs='+', help='Specific folder names to compare in the radar chart (e.g., -c v1 v3 v4)')
    
    args = parser.parse_args()
    
    filenames = os.listdir(args.json_files)
    file_paths = [os.path.join(args.json_files, filename, "metrics.json") for filename in filenames]
    
    df = load_data(file_paths)
    
    if not df.empty:
        create_heatmap(df)
        # Pass the optional args.compare to filter which pipelines show up on the radar chart
        create_radar_chart(df, selected_folders=args.compare)
    else:
        print("No valid data found to plot.")
