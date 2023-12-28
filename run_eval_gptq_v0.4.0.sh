#!/bin/bash

# Llama2-7B GPTQ
# Eval Harness v0.4.0

# Create the 'logs' directory if it does not exist
mkdir -p logs

# Check if an output filename was provided, otherwise use a timestamp
output_filename="logs/${1:-output_gptq_v0.4.0_$(date +%Y%m%d_%H%M%S).log}"

# Define a function to run python with the provided arguments and append output to the specified file
run_python_command() {
    (time lm_eval --model hf \
    --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True \
    --tasks $1 --device cuda:0 --num_fewshot $2 --batch_size 1) |& tee -a "$output_filename"
}

# ARC Challenge
run_python_command "arc_challenge" 25

# HellaSwag
run_python_command  "hellaswag" 10

# MMLU
run_python_command  "mmlu" 5

# TruthfulQA MC
run_python_command "truthfulqa_mc2" 0

# Winogrande
run_python_command "winogrande" 5

# GSM8K
run_python_command "gsm8k" 5

echo "All tasks completed and logged to $output_filename."
