#!/bin/bash

# Check if at least two arguments were provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <harness_version> <gptq_flag> [output_filename]"
    exit 1
fi

# Assign arguments to variables
harness_version="$1"
gptq_flag="$2"
output_filename="${3:-logs/quick_output_${harness_version}_gptq${gptq_flag}_$(date +%Y%m%d_%H%M%S).log}"

# Create the 'logs' directory if it does not exist
mkdir -p logs

# Define a function to run python with the provided arguments and append output to the specified file
run_python_command() {
    if [ "$harness_version" == "v0.4.0" ]; then
        if [ "$gptq_flag" == "True" ]; then
            model_args="pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True"
        else
            model_args="pretrained=meta-llama/Llama-2-7b-chat-hf"
        fi
        (time lm_eval --model hf --model_args $model_args --tasks $1 --device cuda:0 --num_fewshot $2 --batch_size auto) |& tee -a "$output_filename"
    elif [ "$harness_version" == "b281b09" ]; then
        if [ "$gptq_flag" == "True" ]; then
            model_args="pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True"
        else
            model_args="pretrained=meta-llama/Llama-2-7b-chat-hf"
        fi
        (time python ./lm-evaluation-harness/main.py --model hf-causal-experimental --model_args $model_args --tasks $1 --device cuda:0 --num_fewshot $2 --batch_size auto) |& tee -a "$output_filename"
    else
        echo "Invalid harness version."
        exit 1
    fi
}

# Run the common tasks for both versions
run_python_command "arc_challenge" 20 #25
run_python_command "hellaswag" 3 # 10 -> 5 -> 3
run_python_command "winogrande" 5
run_python_command "gsm8k" 2 # 5 -> 3 -> 2


# Run the tasks based on the harness version
if [ "$harness_version" == "v0.4.0" ]; then
    run_python_command "mmlu" 2 # 5
    run_python_command "truthfulqa_mc2" 0
elif [ "$harness_version" == "b281b09" ]; then
    run_python_command "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions" 2
    run_python_command "truthfulqa_mc" 0
fi

echo "All tasks completed and logged to $output_filename."
