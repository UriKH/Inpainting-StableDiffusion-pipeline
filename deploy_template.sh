#!/bin/bash
#SBATCH --job-name=my_remote_job    # A name for your job
#SBATCH --output=result_%j.log     # Standard output and error log (%j inserts job ID)
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --mem=4G                   # Request 4 Gigabytes of RAM
#SBATCH --time=02:00:00            # Time limit (HH:MM:SS)

# --- Your actual commands go below this line ---
echo "Starting my job at: $(date)"
python3 my_script.py               # Replace this with your actual command
echo "Finished at: $(date)"
