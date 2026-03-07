#!/bin/bash
#SBATCH --job-name=testing         # A name for your job
#SBATCH --output=result_%j.log     # Standard output and error log (%j inserts job ID)
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --time=15:00:00            # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:1

# --- Your actual commands go below this line ---
echo "Starting my job at: $(date)"
python ./execution_cmds/executor.py < ./execution_cmds/run_val_0-9_eval_0.txt
echo "Finished at: $(date)"
