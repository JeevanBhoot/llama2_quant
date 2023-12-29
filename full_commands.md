### Llama2 7B
```
lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks arc_challenge --device cuda:0 --num_fewshot 25 --batch_size 1

lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks hellaswag --device cuda:0 --num_fewshot 10 --batch_size 1

lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics,mmlu_computer_security,mmlu_conceptual_physics,mmlu_econometrics,mmlu_electrical_engineering,mmlu_elementary_mathematics,mmlu_formal_logic,mmlu_global_facts,mmlu_high_school_biology,mmlu_high_school_chemistry,mmlu_high_school_computer_science,mmlu_high_school_european_history,mmlu_high_school_geography,mmlu_high_school_government_and_politics,mmlu_high_school_macroeconomics,mmlu_high_school_mathematics,mmlu_high_school_microeconomics,mmlu_high_school_physics,mmlu_high_school_psychology,mmlu_high_school_statistics,mmlu_high_school_us_history,mmlu_high_school_world_history,mmlu_human_aging,mmlu_human_sexuality,mmlu_international_law,mmlu_jurisprudence,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_management,mmlu_marketing,mmlu_medical_genetics,mmlu_miscellaneous,mmlu_moral_disputes,mmlu_moral_scenarios,mmlu_nutrition,mmlu_philosophy,mmlu_prehistory,mmlu_professional_accounting,mmlu_professional_law,mmlu_professional_medicine,mmlu_professional_psychology,mmlu_public_relations,mmlu_security_studies,mmlu_sociology,mmlu_us_foreign_policy,mmlu_virology,mmlu_world_religions --device cuda:0 --num_fewshot 5 --batch_size 1

lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks mmlu --device cuda:0 --num_fewshot 5 --batch_size 1

lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks truthfulqa_mc2 --device cuda:0 --batch_size 1

lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks winogrande --device cuda:0 --num_fewshot 5 --batch_size 1

lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks gsm8k --device cuda:0 --num_fewshot 5 --batch_size 1
```

```
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks hellaswag --device cuda:0 --num_fewshot 10 --batch_size 1

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks truthfulqa_mc --device cuda:0 --num_fewshot 0 --batch_size 1

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks winogrande --device cuda:0 --num_fewshot 5 --batch_size 1

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks gsm8k --device cuda:0 --num_fewshot 5 --batch_size 1
```

### Llama2 7B GPTQ
```
lm_eval --model hf --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,gptq_use_triton=True,load_in_4bit=True --tasks arc_challenge --device cuda:0 --num_fewshot 25 --batch_size 1
```

```
lm_eval --model hf --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True --tasks arc_challenge --device cuda:0 --num_fewshot 25 --batch_size 1

lm_eval --model hf --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True --tasks hellaswag --device cuda:0 --num_fewshot 10 --batch_size 1

lm_eval --model hf --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True --tasks mmlu --device cuda:0 --num_fewshot 5 --batch_size 1

lm_eval --model hf --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True --tasks truthfulqa_mc2 --device cuda:0 --batch_size 1

lm_eval --model hf --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True --tasks winogrande --device cuda:0 --num_fewshot 5 --batch_size 1

lm_eval --model hf --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True --tasks gsm8k --device cuda:0 --num_fewshot 5 --batch_size 1
```

```
python main.py --model hf-causal-experimental --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True --tasks arc_challenge --device cuda:0 --num_fewshot 25 --batch_size 1

python main.py --model hf-causal-experimental --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True --tasks hellaswag --device cuda:0 --num_fewshot 10 --batch_size 1

python main.py --model hf-causal-experimental --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions --device cuda:0 --num_fewshot 5 --batch_size 1

python main.py --model hf-causal-experimental --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True --tasks truthfulqa_mc --device cuda:0 --num_fewshot 0 --batch_size 1
                                              
python main.py --model hf-causal-experimental --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True --tasks winogrande --device cuda:0 --num_fewshot 5 --batch_size 1

python main.py --model hf-causal-experimental --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True --tasks gsm8k --device cuda:0 --num_fewshot 5 --batch_size 1
```