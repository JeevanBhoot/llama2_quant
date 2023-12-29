## Setting Up Miniconda
```
cd ..
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init bash
cd llama2_quant
conda create -n llama2_env python=3.10
conda activate llama2_env
```

## Installing EleutherAI LM Evaluation Harness and AutoGPTQ
Newer version than what HuggingFace uses for Open LLM Leaderboard
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout v0.4.0
# git checkout b281b09
pip install -e .
pip install gekko
# conda install -c "nvidia/label/cuda-11.7.0" cuda
conda install -c "nvidia/label/cuda-12.1.0" cuda
pip install "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.6.0"
# pip uninstall triton
# pip install triton==2.1.0
```

## Llama2 Access
Request access at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Log in with `huggingface-cli login` before running evaluation harness

## Evaluation Commands
On first run, it takes about 10 minutes to download the model weights.

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

## Results
### Llama 2 7B
#### v0.4.0
hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 25, batch_size: 1
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |    25|acc     |0.5026|±  |0.0146|
|             |       |none  |    25|acc_norm|0.5358|±  |0.0146|


real	10m44.376s

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 10, batch_size: 1
|  Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|---------|-------|------|-----:|--------|-----:|---|-----:|
|hellaswag|Yaml   |none  |    10|acc     |0.5927|±  |0.0049|
|         |       |none  |    10|acc_norm|0.7858|±  |0.0041|


real	76m29.362s

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1
|               Tasks               |Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----------------------------------|-------|------|-----:|------|-----:|---|-----:|
|abstract_algebra                   |Yaml   |none  |     5|acc   |0.2900|±  |0.0456|
|anatomy                            |Yaml   |none  |     5|acc   |0.4222|±  |0.0427|
|astronomy                          |Yaml   |none  |     5|acc   |0.4803|±  |0.0407|
|business_ethics                    |Yaml   |none  |     5|acc   |0.5200|±  |0.0502|
|clinical_knowledge                 |Yaml   |none  |     5|acc   |0.5434|±  |0.0307|
|college_biology                    |Yaml   |none  |     5|acc   |0.5139|±  |0.0418|
|college_chemistry                  |Yaml   |none  |     5|acc   |0.2700|±  |0.0446|
|college_computer_science           |Yaml   |none  |     5|acc   |0.3800|±  |0.0488|
|college_mathematics                |Yaml   |none  |     5|acc   |0.3500|±  |0.0479|
|college_medicine                   |Yaml   |none  |     5|acc   |0.3988|±  |0.0373|
|college_physics                    |Yaml   |none  |     5|acc   |0.2255|±  |0.0416|
|computer_security                  |Yaml   |none  |     5|acc   |0.5800|±  |0.0496|
|conceptual_physics                 |Yaml   |none  |     5|acc   |0.4128|±  |0.0322|
|econometrics                       |Yaml   |none  |     5|acc   |0.3684|±  |0.0454|
|electrical_engineering             |Yaml   |none  |     5|acc   |0.5034|±  |0.0417|
|elementary_mathematics             |Yaml   |none  |     5|acc   |0.2989|±  |0.0236|
|formal_logic                       |Yaml   |none  |     5|acc   |0.2540|±  |0.0389|
|global_facts                       |Yaml   |none  |     5|acc   |0.3500|±  |0.0479|
|high_school_biology                |Yaml   |none  |     5|acc   |0.5129|±  |0.0284|
|high_school_chemistry              |Yaml   |none  |     5|acc   |0.3645|±  |0.0339|
|high_school_computer_science       |Yaml   |none  |     5|acc   |0.4100|±  |0.0494|
|high_school_european_history       |Yaml   |none  |     5|acc   |0.5818|±  |0.0385|
|high_school_geography              |Yaml   |none  |     5|acc   |0.5909|±  |0.0350|
|high_school_government_and_politics|Yaml   |none  |     5|acc   |0.6995|±  |0.0331|
|high_school_macroeconomics         |Yaml   |none  |     5|acc   |0.4154|±  |0.0250|
|high_school_mathematics            |Yaml   |none  |     5|acc   |0.2593|±  |0.0267|
|high_school_microeconomics         |Yaml   |none  |     5|acc   |0.4244|±  |0.0321|
|high_school_physics                |Yaml   |none  |     5|acc   |0.2848|±  |0.0368|
|high_school_psychology             |Yaml   |none  |     5|acc   |0.6752|±  |0.0201|
|high_school_statistics             |Yaml   |none  |     5|acc   |0.3380|±  |0.0323|
|high_school_us_history             |Yaml   |none  |     5|acc   |0.6765|±  |0.0328|
|high_school_world_history          |Yaml   |none  |     5|acc   |0.6709|±  |0.0306|
|human_aging                        |Yaml   |none  |     5|acc   |0.5740|±  |0.0332|
|human_sexuality                    |Yaml   |none  |     5|acc   |0.5725|±  |0.0434|
|international_law                  |Yaml   |none  |     5|acc   |0.6364|±  |0.0439|
|jurisprudence                      |Yaml   |none  |     5|acc   |0.5926|±  |0.0475|
|logical_fallacies                  |Yaml   |none  |     5|acc   |0.5399|±  |0.0392|
|machine_learning                   |Yaml   |none  |     5|acc   |0.3036|±  |0.0436|
|management                         |Yaml   |none  |     5|acc   |0.6699|±  |0.0466|
|marketing                          |Yaml   |none  |     5|acc   |0.7094|±  |0.0297|
|medical_genetics                   |Yaml   |none  |     5|acc   |0.4900|±  |0.0502|
|miscellaneous                      |Yaml   |none  |     5|acc   |0.6782|±  |0.0167|
|moral_disputes                     |Yaml   |none  |     5|acc   |0.5231|±  |0.0269|
|moral_scenarios                    |Yaml   |none  |     5|acc   |0.2313|±  |0.0141|
|nutrition                          |Yaml   |none  |     5|acc   |0.5196|±  |0.0286|
|philosophy                         |Yaml   |none  |     5|acc   |0.5595|±  |0.0282|
|prehistory                         |Yaml   |none  |     5|acc   |0.5679|±  |0.0276|
|professional_accounting            |Yaml   |none  |     5|acc   |0.3617|±  |0.0287|
|professional_law                   |Yaml   |none  |     5|acc   |0.3501|±  |0.0122|
|professional_medicine              |Yaml   |none  |     5|acc   |0.4596|±  |0.0303|
|professional_psychology            |Yaml   |none  |     5|acc   |0.4788|±  |0.0202|
|public_relations                   |Yaml   |none  |     5|acc   |0.5364|±  |0.0478|
|security_studies                   |Yaml   |none  |     5|acc   |0.5265|±  |0.0320|
|sociology                          |Yaml   |none  |     5|acc   |0.6468|±  |0.0338|
|us_foreign_policy                  |Yaml   |none  |     5|acc   |0.7100|±  |0.0456|
|virology                           |Yaml   |none  |     5|acc   |0.4277|±  |0.0385|
|world_religions                    |Yaml   |none  |     5|acc   |0.7251|±  |0.0342|

avg acc: 27.4563/57 = 0.4817% vs 48.32% (hf) //
1 hr 27 mins 16 secs (hypatia)

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1    
|      Groups      |Version|Filter|n-shot|Metric|Value |   |Stderr|
|------------------|-------|------|-----:|------|-----:|---|-----:|
|mmlu              |N/A    |none  |     0|acc   |0.4724|±  |0.1260|
| - humanities     |N/A    |none  |     5|acc   |0.4380|±  |0.1478|
| - other          |N/A    |none  |     5|acc   |0.5472|±  |0.0981|
| - social_sciences|N/A    |none  |     5|acc   |0.5466|±  |0.0945|
| - stem           |N/A    |none  |     5|acc   |0.3777|±  |0.0940|


real	92m0.077s

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 0, batch_size: 1
|    Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|--------------|-------|------|-----:|------|-----:|---|-----:|
|truthfulqa_mc2|Yaml   |none  |     0|acc   |0.4531|±  |0.0156|


real	2m56.960s

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1
|  Tasks   |Version|Filter|n-shot|Metric|Value |   |Stderr|
|----------|-------|------|-----:|------|-----:|---|-----:|
|winogrande|Yaml   |none  |     5|acc   |0.6638|±  |0.0133|


real	0m59.503s

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|
|-----|-------|----------|-----:|-----------|-----:|---|-----:|
|gsm8k|Yaml   |get-answer|     5|exact_match|0.2274|±  |0.0115|


real	54m50.767s


#### HF version (commit b281b09)
hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf), limit: None, provide_description: False, num_fewshot: 25, batch_size: 1
|    Task     |Version| Metric |Value |   |Stderr|
|-------------|------:|--------|-----:|---|-----:|
|arc_challenge|      0|acc     |0.4940|±  |0.0146|
|             |       |acc_norm|0.5273|±  |0.0146|

acc_norm: 52.73%
real	13m17.961s

|  Task   |Version| Metric |Value |   |Stderr|
|---------|------:|--------|-----:|---|-----:|
|hellaswag|      0|acc     |0.5968|±  |0.0049|
|         |       |acc_norm|0.7852|±  |0.0041|

acc_norm: 78.52%
real	98m58.470s

hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf), limit: None, provide_description: False, num_fewshot: 5, batch_size: 1
|                      Task                       |Version| Metric |Value |   |Stderr|
|-------------------------------------------------|------:|--------|-----:|---|-----:|
|hendrycksTest-abstract_algebra                   |      1|acc     |0.2900|±  |0.0456|
|                                                 |       |acc_norm|0.2900|±  |0.0456|
|hendrycksTest-anatomy                            |      1|acc     |0.4222|±  |0.0427|
|                                                 |       |acc_norm|0.4222|±  |0.0427|
|hendrycksTest-astronomy                          |      1|acc     |0.4803|±  |0.0407|
|                                                 |       |acc_norm|0.4803|±  |0.0407|
|hendrycksTest-business_ethics                    |      1|acc     |0.5200|±  |0.0502|
|                                                 |       |acc_norm|0.5200|±  |0.0502|
|hendrycksTest-clinical_knowledge                 |      1|acc     |0.5434|±  |0.0307|
|                                                 |       |acc_norm|0.5434|±  |0.0307|
|hendrycksTest-college_biology                    |      1|acc     |0.5139|±  |0.0418|
|                                                 |       |acc_norm|0.5139|±  |0.0418|
|hendrycksTest-college_chemistry                  |      1|acc     |0.2700|±  |0.0446|
|                                                 |       |acc_norm|0.2700|±  |0.0446|
|hendrycksTest-college_computer_science           |      1|acc     |0.3800|±  |0.0488|
|                                                 |       |acc_norm|0.3800|±  |0.0488|
|hendrycksTest-college_mathematics                |      1|acc     |0.3500|±  |0.0479|
|                                                 |       |acc_norm|0.3500|±  |0.0479|
|hendrycksTest-college_medicine                   |      1|acc     |0.3988|±  |0.0373|
|                                                 |       |acc_norm|0.3988|±  |0.0373|
|hendrycksTest-college_physics                    |      1|acc     |0.2255|±  |0.0416|
|                                                 |       |acc_norm|0.2255|±  |0.0416|
|hendrycksTest-computer_security                  |      1|acc     |0.5800|±  |0.0496|
|                                                 |       |acc_norm|0.5800|±  |0.0496|
|hendrycksTest-conceptual_physics                 |      1|acc     |0.4128|±  |0.0322|
|                                                 |       |acc_norm|0.4128|±  |0.0322|
|hendrycksTest-econometrics                       |      1|acc     |0.3684|±  |0.0454|
|                                                 |       |acc_norm|0.3684|±  |0.0454|
|hendrycksTest-electrical_engineering             |      1|acc     |0.5034|±  |0.0417|
|                                                 |       |acc_norm|0.5034|±  |0.0417|
|hendrycksTest-elementary_mathematics             |      1|acc     |0.2989|±  |0.0236|
|                                                 |       |acc_norm|0.2989|±  |0.0236|
|hendrycksTest-formal_logic                       |      1|acc     |0.2540|±  |0.0389|
|                                                 |       |acc_norm|0.2540|±  |0.0389|
|hendrycksTest-global_facts                       |      1|acc     |0.3500|±  |0.0479|
|                                                 |       |acc_norm|0.3500|±  |0.0479|
|hendrycksTest-high_school_biology                |      1|acc     |0.5129|±  |0.0284|
|                                                 |       |acc_norm|0.5129|±  |0.0284|
|hendrycksTest-high_school_chemistry              |      1|acc     |0.3645|±  |0.0339|
|                                                 |       |acc_norm|0.3645|±  |0.0339|
|hendrycksTest-high_school_computer_science       |      1|acc     |0.4100|±  |0.0494|
|                                                 |       |acc_norm|0.4100|±  |0.0494|
|hendrycksTest-high_school_european_history       |      1|acc     |0.5818|±  |0.0385|
|                                                 |       |acc_norm|0.5818|±  |0.0385|
|hendrycksTest-high_school_geography              |      1|acc     |0.5909|±  |0.0350|
|                                                 |       |acc_norm|0.5909|±  |0.0350|
|hendrycksTest-high_school_government_and_politics|      1|acc     |0.6995|±  |0.0331|
|                                                 |       |acc_norm|0.6995|±  |0.0331|
|hendrycksTest-high_school_macroeconomics         |      1|acc     |0.4154|±  |0.0250|
|                                                 |       |acc_norm|0.4154|±  |0.0250|
|hendrycksTest-high_school_mathematics            |      1|acc     |0.2593|±  |0.0267|
|                                                 |       |acc_norm|0.2593|±  |0.0267|
|hendrycksTest-high_school_microeconomics         |      1|acc     |0.4244|±  |0.0321|
|                                                 |       |acc_norm|0.4244|±  |0.0321|
|hendrycksTest-high_school_physics                |      1|acc     |0.2848|±  |0.0368|
|                                                 |       |acc_norm|0.2848|±  |0.0368|
|hendrycksTest-high_school_psychology             |      1|acc     |0.6752|±  |0.0201|
|                                                 |       |acc_norm|0.6752|±  |0.0201|
|hendrycksTest-high_school_statistics             |      1|acc     |0.3380|±  |0.0323|
|                                                 |       |acc_norm|0.3380|±  |0.0323|
|hendrycksTest-high_school_us_history             |      1|acc     |0.6765|±  |0.0328|
|                                                 |       |acc_norm|0.6765|±  |0.0328|
|hendrycksTest-high_school_world_history          |      1|acc     |0.6709|±  |0.0306|
|                                                 |       |acc_norm|0.6709|±  |0.0306|
|hendrycksTest-human_aging                        |      1|acc     |0.5740|±  |0.0332|
|                                                 |       |acc_norm|0.5740|±  |0.0332|
|hendrycksTest-human_sexuality                    |      1|acc     |0.5725|±  |0.0434|
|                                                 |       |acc_norm|0.5725|±  |0.0434|
|hendrycksTest-international_law                  |      1|acc     |0.6364|±  |0.0439|
|                                                 |       |acc_norm|0.6364|±  |0.0439|
|hendrycksTest-jurisprudence                      |      1|acc     |0.5926|±  |0.0475|
|                                                 |       |acc_norm|0.5926|±  |0.0475|
|hendrycksTest-logical_fallacies                  |      1|acc     |0.5399|±  |0.0392|
|                                                 |       |acc_norm|0.5399|±  |0.0392|
|hendrycksTest-machine_learning                   |      1|acc     |0.3036|±  |0.0436|
|                                                 |       |acc_norm|0.3036|±  |0.0436|
|hendrycksTest-management                         |      1|acc     |0.6699|±  |0.0466|
|                                                 |       |acc_norm|0.6699|±  |0.0466|
|hendrycksTest-marketing                          |      1|acc     |0.7094|±  |0.0297|
|                                                 |       |acc_norm|0.7094|±  |0.0297|
|hendrycksTest-medical_genetics                   |      1|acc     |0.4900|±  |0.0502|
|                                                 |       |acc_norm|0.4900|±  |0.0502|
|hendrycksTest-miscellaneous                      |      1|acc     |0.6782|±  |0.0167|
|                                                 |       |acc_norm|0.6782|±  |0.0167|
|hendrycksTest-moral_disputes                     |      1|acc     |0.5231|±  |0.0269|
|                                                 |       |acc_norm|0.5231|±  |0.0269|
|hendrycksTest-moral_scenarios                    |      1|acc     |0.2313|±  |0.0141|
|                                                 |       |acc_norm|0.2313|±  |0.0141|
|hendrycksTest-nutrition                          |      1|acc     |0.5196|±  |0.0286|
|                                                 |       |acc_norm|0.5196|±  |0.0286|
|hendrycksTest-philosophy                         |      1|acc     |0.5595|±  |0.0282|
|                                                 |       |acc_norm|0.5595|±  |0.0282|
|hendrycksTest-prehistory                         |      1|acc     |0.5679|±  |0.0276|
|                                                 |       |acc_norm|0.5679|±  |0.0276|
|hendrycksTest-professional_accounting            |      1|acc     |0.3617|±  |0.0287|
|                                                 |       |acc_norm|0.3617|±  |0.0287|
|hendrycksTest-professional_law                   |      1|acc     |0.3501|±  |0.0122|
|                                                 |       |acc_norm|0.3501|±  |0.0122|
|hendrycksTest-professional_medicine              |      1|acc     |0.4596|±  |0.0303|
|                                                 |       |acc_norm|0.4596|±  |0.0303|
|hendrycksTest-professional_psychology            |      1|acc     |0.4788|±  |0.0202|
|                                                 |       |acc_norm|0.4788|±  |0.0202|
|hendrycksTest-public_relations                   |      1|acc     |0.5364|±  |0.0478|
|                                                 |       |acc_norm|0.5364|±  |0.0478|
|hendrycksTest-security_studies                   |      1|acc     |0.5265|±  |0.0320|
|                                                 |       |acc_norm|0.5265|±  |0.0320|
|hendrycksTest-sociology                          |      1|acc     |0.6468|±  |0.0338|
|                                                 |       |acc_norm|0.6468|±  |0.0338|
|hendrycksTest-us_foreign_policy                  |      1|acc     |0.7100|±  |0.0456|
|                                                 |       |acc_norm|0.7100|±  |0.0456|
|hendrycksTest-virology                           |      1|acc     |0.4277|±  |0.0385|
|                                                 |       |acc_norm|0.4277|±  |0.0385|
|hendrycksTest-world_religions                    |      1|acc     |0.7251|±  |0.0342|
|                                                 |       |acc_norm|0.7251|±  |0.0342|


real	118m34.893s

hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf), limit: None, provide_description: False, num_fewshot: 0, batch_size: 1
|    Task     |Version|Metric|Value |   |Stderr|
|-------------|------:|------|-----:|---|-----:|
|truthfulqa_mc|      1|mc1   |0.3011|±  |0.0161|
|             |       |mc2   |0.4531|±  |0.0156|

mc2 acc: 45.31%
real	3m9.774s


hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf), limit: None, provide_description: False, num_fewshot: 5, batch_size: 1
|   Task   |Version|Metric|Value |   |Stderr|
|----------|------:|------|-----:|---|-----:|
|winogrande|      0|acc   |0.7301|±  |0.0125|

acc: 73.01%
real	1m27.888s


hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf), limit: None, provide_description: False, num_fewshot: 5, batch_size: 1
|Task |Version|Metric|Value |   |Stderr|
|-----|------:|------|-----:|---|-----:|
|gsm8k|      0|acc   |0.1873|±  |0.0107|

acc: 18.73%
real	48m6.565s

### Llama2 7B GPTQ
#### v0.4.0 with Triton
hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,gptq_use_triton=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 25, batch_size: 1
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |    25|acc     |0.4718|±  |0.0146|
|             |       |none  |    25|acc_norm|0.4949|±  |0.0146|

acc_norm: 49.49% //
18 mins 20 secs

#### v0.4.0 without Triton
hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 25, batch_size: 1
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |    25|acc     |0.4744|±  |0.0146|
|             |       |none  |    25|acc_norm|0.4949|±  |0.0146|


real	9m2.280s

hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 10, batch_size: 1
|  Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|---------|-------|------|-----:|--------|-----:|---|-----:|
|hellaswag|Yaml   |none  |    10|acc     |0.5488|±  |0.0050|
|         |       |none  |    10|acc_norm|0.7245|±  |0.0045|


real	63m47.118s

hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1 
|      Groups      |Version|Filter|n-shot|Metric|Value |   |Stderr|
|------------------|-------|------|-----:|------|-----:|---|-----:|
|mmlu              |N/A    |none  |     0|acc   |0.4311|±  |0.1268|
| - humanities     |N/A    |none  |     5|acc   |0.3781|±  |0.1567|
| - other          |N/A    |none  |     5|acc   |0.5146|±  |0.0993|
| - social_sciences|N/A    |none  |     5|acc   |0.5018|±  |0.0855|
| - stem           |N/A    |none  |     5|acc   |0.3587|±  |0.0913|


real	80m22.880s

hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 0, batch_size: 1
|    Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|--------------|-------|------|-----:|------|-----:|---|-----:|
|truthfulqa_mc2|Yaml   |none  |     0|acc   |0.4411|±  |0.0156|


real	4m11.465s

hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1
|  Tasks   |Version|Filter|n-shot|Metric|Value |   |Stderr|
|----------|-------|------|-----:|------|-----:|---|-----:|
|winogrande|Yaml   |none  |     5|acc   |0.6543|±  |0.0134|


real	1m0.571s

hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|
|-----|-------|----------|-----:|-----------|-----:|---|-----:|
|gsm8k|Yaml   |get-answer|     5|exact_match|0.1524|±  |0.0099|


real	30m21.317s

#### HF Version (commit b281b09)
hf-causal-experimental (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True), limit: None, provide_description: False, num_fewshot: 25, batch_size: 1
|    Task     |Version| Metric |Value |   |Stderr|
|-------------|------:|--------|-----:|---|-----:|
|arc_challenge|      0|acc     |0.4881|±  |0.0146|
|             |       |acc_norm|0.5128|±  |0.0146|

acc_norm: 51.28%
real	11m28.755s

|  Task   |Version| Metric |Value |   |Stderr|
|---------|------:|--------|-----:|---|-----:|
|hellaswag|      0|acc     |0.5474|±  |0.0050|
|         |       |acc_norm|0.7201|±  |0.0045|

acc_norm: 72.01%
real	86m12.379s

hf-causal-experimental (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 1
|                      Task                       |Version| Metric |Value |   |Stderr|
|-------------------------------------------------|------:|--------|-----:|---|-----:|
|hendrycksTest-abstract_algebra                   |      1|acc     |0.3000|±  |0.0461|
|                                                 |       |acc_norm|0.3000|±  |0.0461|
|hendrycksTest-anatomy                            |      1|acc     |0.4370|±  |0.0428|
|                                                 |       |acc_norm|0.4370|±  |0.0428|
|hendrycksTest-astronomy                          |      1|acc     |0.4539|±  |0.0405|
|                                                 |       |acc_norm|0.4539|±  |0.0405|
|hendrycksTest-business_ethics                    |      1|acc     |0.3400|±  |0.0476|
|                                                 |       |acc_norm|0.3400|±  |0.0476|
|hendrycksTest-clinical_knowledge                 |      1|acc     |0.4868|±  |0.0308|
|                                                 |       |acc_norm|0.4868|±  |0.0308|
|hendrycksTest-college_biology                    |      1|acc     |0.4722|±  |0.0417|
|                                                 |       |acc_norm|0.4722|±  |0.0417|
|hendrycksTest-college_chemistry                  |      1|acc     |0.2300|±  |0.0423|
|                                                 |       |acc_norm|0.2300|±  |0.0423|
|hendrycksTest-college_computer_science           |      1|acc     |0.3500|±  |0.0479|
|                                                 |       |acc_norm|0.3500|±  |0.0479|
|hendrycksTest-college_mathematics                |      1|acc     |0.3000|±  |0.0461|
|                                                 |       |acc_norm|0.3000|±  |0.0461|
|hendrycksTest-college_medicine                   |      1|acc     |0.3815|±  |0.0370|
|                                                 |       |acc_norm|0.3815|±  |0.0370|
|hendrycksTest-college_physics                    |      1|acc     |0.2157|±  |0.0409|
|                                                 |       |acc_norm|0.2157|±  |0.0409|
|hendrycksTest-computer_security                  |      1|acc     |0.5400|±  |0.0501|
|                                                 |       |acc_norm|0.5400|±  |0.0501|
|hendrycksTest-conceptual_physics                 |      1|acc     |0.3915|±  |0.0319|
|                                                 |       |acc_norm|0.3915|±  |0.0319|
|hendrycksTest-econometrics                       |      1|acc     |0.3333|±  |0.0443|
|                                                 |       |acc_norm|0.3333|±  |0.0443|
|hendrycksTest-electrical_engineering             |      1|acc     |0.4069|±  |0.0409|
|                                                 |       |acc_norm|0.4069|±  |0.0409|
|hendrycksTest-elementary_mathematics             |      1|acc     |0.2751|±  |0.0230|
|                                                 |       |acc_norm|0.2751|±  |0.0230|
|hendrycksTest-formal_logic                       |      1|acc     |0.2302|±  |0.0376|
|                                                 |       |acc_norm|0.2302|±  |0.0376|
|hendrycksTest-global_facts                       |      1|acc     |0.3700|±  |0.0485|
|                                                 |       |acc_norm|0.3700|±  |0.0485|
|hendrycksTest-high_school_biology                |      1|acc     |0.5194|±  |0.0284|
|                                                 |       |acc_norm|0.5194|±  |0.0284|
|hendrycksTest-high_school_chemistry              |      1|acc     |0.3251|±  |0.0330|
|                                                 |       |acc_norm|0.3251|±  |0.0330|
|hendrycksTest-high_school_computer_science       |      1|acc     |0.4100|±  |0.0494|
|                                                 |       |acc_norm|0.4100|±  |0.0494|
|hendrycksTest-high_school_european_history       |      1|acc     |0.2364|±  |0.0332|
|                                                 |       |acc_norm|0.2364|±  |0.0332|
|hendrycksTest-high_school_geography              |      1|acc     |0.5808|±  |0.0352|
|                                                 |       |acc_norm|0.5808|±  |0.0352|
|hendrycksTest-high_school_government_and_politics|      1|acc     |0.6269|±  |0.0349|
|                                                 |       |acc_norm|0.6269|±  |0.0349|
|hendrycksTest-high_school_macroeconomics         |      1|acc     |0.3795|±  |0.0246|
|                                                 |       |acc_norm|0.3795|±  |0.0246|
|hendrycksTest-high_school_mathematics            |      1|acc     |0.2481|±  |0.0263|
|                                                 |       |acc_norm|0.2481|±  |0.0263|
|hendrycksTest-high_school_microeconomics         |      1|acc     |0.3950|±  |0.0318|
|                                                 |       |acc_norm|0.3950|±  |0.0318|
|hendrycksTest-high_school_physics                |      1|acc     |0.2980|±  |0.0373|
|                                                 |       |acc_norm|0.2980|±  |0.0373|
|hendrycksTest-high_school_psychology             |      1|acc     |0.6147|±  |0.0209|
|                                                 |       |acc_norm|0.6147|±  |0.0209|
|hendrycksTest-high_school_statistics             |      1|acc     |0.3102|±  |0.0315|
|                                                 |       |acc_norm|0.3102|±  |0.0315|
|hendrycksTest-high_school_us_history             |      1|acc     |0.6176|±  |0.0341|
|                                                 |       |acc_norm|0.6176|±  |0.0341|
|hendrycksTest-high_school_world_history          |      1|acc     |0.6287|±  |0.0315|
|                                                 |       |acc_norm|0.6287|±  |0.0315|
|hendrycksTest-human_aging                        |      1|acc     |0.5695|±  |0.0332|
|                                                 |       |acc_norm|0.5695|±  |0.0332|
|hendrycksTest-human_sexuality                    |      1|acc     |0.4885|±  |0.0438|
|                                                 |       |acc_norm|0.4885|±  |0.0438|
|hendrycksTest-international_law                  |      1|acc     |0.5868|±  |0.0450|
|                                                 |       |acc_norm|0.5868|±  |0.0450|
|hendrycksTest-jurisprudence                      |      1|acc     |0.5648|±  |0.0479|
|                                                 |       |acc_norm|0.5648|±  |0.0479|
|hendrycksTest-logical_fallacies                  |      1|acc     |0.4724|±  |0.0392|
|                                                 |       |acc_norm|0.4724|±  |0.0392|
|hendrycksTest-machine_learning                   |      1|acc     |0.3482|±  |0.0452|
|                                                 |       |acc_norm|0.3482|±  |0.0452|
|hendrycksTest-management                         |      1|acc     |0.5728|±  |0.0490|
|                                                 |       |acc_norm|0.5728|±  |0.0490|
|hendrycksTest-marketing                          |      1|acc     |0.6880|±  |0.0304|
|                                                 |       |acc_norm|0.6880|±  |0.0304|
|hendrycksTest-medical_genetics                   |      1|acc     |0.4400|±  |0.0499|
|                                                 |       |acc_norm|0.4400|±  |0.0499|
|hendrycksTest-miscellaneous                      |      1|acc     |0.6564|±  |0.0170|
|                                                 |       |acc_norm|0.6564|±  |0.0170|
|hendrycksTest-moral_disputes                     |      1|acc     |0.4913|±  |0.0269|
|                                                 |       |acc_norm|0.4913|±  |0.0269|
|hendrycksTest-moral_scenarios                    |      1|acc     |0.2447|±  |0.0144|
|                                                 |       |acc_norm|0.2447|±  |0.0144|
|hendrycksTest-nutrition                          |      1|acc     |0.4837|±  |0.0286|
|                                                 |       |acc_norm|0.4837|±  |0.0286|
|hendrycksTest-philosophy                         |      1|acc     |0.5627|±  |0.0282|
|                                                 |       |acc_norm|0.5627|±  |0.0282|
|hendrycksTest-prehistory                         |      1|acc     |0.5556|±  |0.0276|
|                                                 |       |acc_norm|0.5556|±  |0.0276|
|hendrycksTest-professional_accounting            |      1|acc     |0.3582|±  |0.0286|
|                                                 |       |acc_norm|0.3582|±  |0.0286|
|hendrycksTest-professional_law                   |      1|acc     |0.2379|±  |0.0109|
|                                                 |       |acc_norm|0.2379|±  |0.0109|
|hendrycksTest-professional_medicine              |      1|acc     |0.4154|±  |0.0299|
|                                                 |       |acc_norm|0.4154|±  |0.0299|
|hendrycksTest-professional_psychology            |      1|acc     |0.4542|±  |0.0201|
|                                                 |       |acc_norm|0.4542|±  |0.0201|
|hendrycksTest-public_relations                   |      1|acc     |0.5182|±  |0.0479|
|                                                 |       |acc_norm|0.5182|±  |0.0479|
|hendrycksTest-security_studies                   |      1|acc     |0.4653|±  |0.0319|
|                                                 |       |acc_norm|0.4653|±  |0.0319|
|hendrycksTest-sociology                          |      1|acc     |0.5672|±  |0.0350|
|                                                 |       |acc_norm|0.5672|±  |0.0350|
|hendrycksTest-us_foreign_policy                  |      1|acc     |0.6600|±  |0.0476|
|                                                 |       |acc_norm|0.6600|±  |0.0476|
|hendrycksTest-virology                           |      1|acc     |0.3976|±  |0.0381|
|                                                 |       |acc_norm|0.3976|±  |0.0381|
|hendrycksTest-world_religions                    |      1|acc     |0.6901|±  |0.0355|
|                                                 |       |acc_norm|0.6901|±  |0.0355|

avg acc = 25.194/57 = 44.2%
real	106m48.030s

hf-causal-experimental (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True), limit: None, provide_description: False, num_fewshot: 0, batch_size: 1
|    Task     |Version|Metric|Value |   |Stderr|
|-------------|------:|------|-----:|---|-----:|
|truthfulqa_mc|      1|mc1   |0.2901|±  |0.0159|
|             |       |mc2   |0.4411|±  |0.0156|

mc2 acc: 44.11%
real	4m28.666s

hf-causal-experimental (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 1
|   Task   |Version|Metric|Value|   |Stderr|
|----------|------:|------|----:|---|-----:|
|winogrande|      0|acc   |0.708|±  |0.0128|

acc: 70.8%
real	2m4.967s

hf-causal-experimental (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,quantized=model.safetensors,load_in_4bit=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 1
|Task |Version|Metric|Value |   |Stderr|
|-----|------:|------|-----:|---|-----:|
|gsm8k|      0|acc   |0.1312|±  |0.0093|

acc: 13.12%
real	29m29.713s

### HuggingFace Open LLM Leaderboard
| Model                 | Average | ARC  | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K |
|-----------------------|---------|------|-----------|------|------------|------------|-------|
| meta-llama/llama-2-7b-chat-hf | 50.74    | 52.9 | 78.55     | 48.32 | 45.57       | 71.74      | 7.35  |
| TheBloke/Llama-2-7B-GPTQ         | 48.48   | 52.05| 77.59     | 43.99| 39.32       | 72.93      | 5     |
| TheBloke/Llama-2-7b-Chat-AWQ     | 29.14   | 27.22| 25.48     | 24.67| 49.95       | 47.51      | 0     |


| Model                 |Harness Version| Average | ARC  | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K |
|-----------------------|----------|---------|------|-----------|------|------------|------------|-------|
| meta-llama/llama-2-7b-chat-hf | v0.4.0   | 52.31 | 53.58 | 78.58  | 47.24* | 45.31  | 66.38  | 22.74** |
| meta-llama/llama-2-7b-chat-hf | b281b09  |       | 52.3  | 78.52  |        | 45.31  | 73.01  | 18.73   |
| TheBloke/Llama-2-7B-Chat-GPTQ | v0.4.0   | 48.58 | 49.49 | 74.25  | 43.11* | 44.11  | 65.43  | 15.24** |
| TheBloke/Llama-2-7B-Chat-GPTQ | b281b09  | 49.25 | 51.28 | 72.01  | 44.20  | 44.11  | 70.80  | 13.12   |

*weighted (over num samples) average (vs unweighted avg)
**exact match metric (vs average)
v0.4.0 Winogrande - new preprocessing?

Time taken to evaluate:

| Model                            | Harness Version | ARC  | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K | Total |
|----------------------------------|-----------------|------|-----------|------|------------|------------|-------|-------|
| meta-llama/llama-2-7b-chat-hf    | v0.4.0          | 10m44s | 1h16m29s | 1h32m00s | 2m57s | 1m00s | 54m51s | 3h58m |
| meta-llama/llama-2-7b-chat-hf    | b281b09         | 13m18s | 1h38m58s | 1h58m35s | 3m10s | 1m28s | 48m07s | 4h43m |
| TheBloke/Llama-2-7B-Chat-GPTQ    | v0.4.0          | 9m02s  | 1h03m47s | 1h20m23s | 4m11s | 1m01  | 30m21s | 3h09m |
| TheBloke/Llama-2-7B-Chat-GPTQ    | b281b09         | 11m29s | 1h26m12s | 1h46m48s | 4m29s | 2m05s | 29m30s | 4h00m |
