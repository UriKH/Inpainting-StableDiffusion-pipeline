import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def create_graphs(file_paths):
    # 2. Extract and structure the data
    data = []
    for path in file_paths:
        # Get the name of the folder containing the JSON file
        folder_name = os.path.basename(os.path.dirname(path))
        
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        with open(path, 'r') as f:
            metrics_dict = json.load(f)
            
            # Flatten into a list of records for pandas
            for metric_name, score in metrics_dict.items():
                if isinstance(score, (int, float)):
                    data.append({
                        "Folder": folder_name,
                        "Metric": metric_name,
                        "Score": score
                    })

    # Create a DataFrame
    df = pd.DataFrame(data)

    # 3. Graphing setup
    # Use seaborn's whitegrid style for a clean, modern look
    sns.set_theme(style="whitegrid") 

    # Get a list of all unique metrics found across all JSON files
    unique_metrics = df['Metric'].unique()

    sorted_folders = sorted(df['Folder'].unique())

    # 4. Generate a graph for each metric
    for metric in unique_metrics:
        # Filter the dataframe for only the current metric
        metric_df = df[df['Metric'] == metric]
        
        plt.figure(figsize=(8, 5))
        
        # Create the bar plot
        ax = sns.barplot(
            data=metric_df, 
            x="Folder", 
            y="Score", 
            hue="Folder",      # Assigns different colors to different folders
            palette="viridis", # A nice, colorblind-friendly color palette
            legend=False,
            order=sorted_folders
        )
        
        # Aesthetics
        plt.title(f"Comparison of {metric}", fontsize=14, fontweight='bold')
        plt.xlabel("Source Folder", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        
        # Rotate the x-axis labels in case your folder names are long
        plt.xticks(rotation=45, ha="right") 
        
        # Adjust layout so labels don't get cut off
        plt.tight_layout()
        
        # Show the plot (or use plt.savefig(f"{metric}_chart.png") to save them)
        plt.savefig(f"{metric}_chart.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate graphs from JSON metric files.")
    parser.add_argument('-j', '--json-files', help='Paths to the global directory in which all the directories which contain metrics JSON files.')
    
    args = parser.parse_args()
    
    filenames = os.listdir(args.json_files)
    file_paths = [os.path.join(args.json_files, filename, "metrics.json") for filename in filenames]
    create_graphs(file_paths)