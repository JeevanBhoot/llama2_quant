#!/bin/bash

# Llama2-7B
# HF Eval Harness Version (commit b281b09)

# Create the 'logs' directory if it does not exist
mkdir -p logs

# Check if an output filename was provided, otherwise use a timestamp
output_filename="logs/${1:-output_$(date +%Y%m%d_%H%M%S).log}"

# Define a function to run python with the provided arguments and append output to the specified file
run_python_command() {
    python ./lm-evaluation-harness/main.py --model hf-causal-experimental \
    --model_args pretrained=meta-llama/Llama-2-7b-chat-hf \
    --tasks $1 --device cuda:0 --num_fewshot $2 --batch_size 1 | tee -a "$output_filename"
}

# ARC Challenge
run_python_command "arc_challenge" 25

# HellaSwag
run_python_command  "hellaswag" 10

# HendrycksTest (Multiple Tasks)
run_python_command "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions" 5

# TruthfulQA MC
run_python_command "truthfulqa_mc" 0

# Winogrande
run_python_command "winogrande" 5

# GSM8K
run_python_command "gsm8k" 5

echo "All tasks completed and logged to $output_filename."
