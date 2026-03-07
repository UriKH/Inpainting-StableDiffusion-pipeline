#!/bin/bash
#SBATCH --job-name=my_remote_job    # A name for your job
#SBATCH --output=result_%j.log     # Standard output and error log (%j inserts job ID)
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --time=12:00:00            # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:1 		   # Ask for GPU

echo "Starting my job at: $(date)"
python3 execution_cmds/executor.py < YOU_COMMANDS_FILE               # Replace this with your actual command
echo "Finished at: $(date)"
