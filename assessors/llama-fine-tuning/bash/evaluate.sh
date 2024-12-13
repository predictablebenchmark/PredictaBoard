#!/bin/bash
#SBATCH --job-name=pb-ev
#SBATCH --output=../logs/mb_array_%A_%a.out
#SBATCH --error=../logs/mb_array_%A_%a.err
#SBATCH --array=1-5:1

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH 
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --constraint=a100_80gb
#SBATCH --nice=1000

# Define an array of input files
models=("assessors/llama-fine-tuning/models/Llama-3.2-1B-Instruct-bnb-4bit_meta-llama__Meta-Llama-3.1-70B-Instruct_mmlu_pro_train_set/checkpoint-2130" "assessors/llama-fine-tuning/models/Llama-3.2-1B-Instruct-bnb-4bit_migtissera__Tess-3-Mistral-Nemo-12B_mmlu_pro_train_set/checkpoint-2130" "assessors/llama-fine-tuning/models/Llama-3.2-1B-Instruct-bnb-4bit_OpenAI__GPT-4o-2024-05-13_mmlu_pro_train_set/checkpoint-2130" "assessors/llama-fine-tuning/models/Llama-3.2-1B-Instruct-bnb-4bit_OpenAI__GPT-4o-2024-08-06_mmlu_pro_train_set/checkpoint-2130" "assessors/llama-fine-tuning/models/Llama-3.2-1B-Instruct-bnb-4bit_OpenAI__GPT-4o-mini_mmlu_pro_train_set/checkpoint-2130")

# Calculate the index of the current task in the array
task_index=$((SLURM_ARRAY_TASK_ID - 1))

# Check if the current task is within the range of input files
if [ "$task_index" -lt "${#models[@]}" ]; then
    # Extract the input file for the current task
    current_input=${models[$task_index]}

    # Print task information
    echo "Running task $SLURM_ARRAY_TASK_ID with model: $current_input"

    cd ~/github-repos/predbench_project/predbench

    export PATH="~/miniconda3/envs/unsloth_env/bin:$PATH"

    source activate unsloth_env

    python ./assessors/llama-fine-tuning/evaluate.py --checkpoint $current_input

    echo "Task $SLURM_ARRAY_TASK_ID completed"
else
    echo "No task to run for task index $task_index"

fi