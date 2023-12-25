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
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True --tasks hellaswag --device cuda:0 --num_fewshot 10 --batch_size 8

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True --tasks truthfulqa_mc --device cuda:0 --num_fewshot 0 --batch_size 8

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True --tasks winogrande --device cuda:0 --num_fewshot 5 --batch_size 8

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True --tasks gsm8k --device cuda:0 --num_fewshot 5 --batch_size 4
```

### Llama2 7B GPTQ
```
lm_eval --model hf --model_args pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,gptq_use_triton=True,load_in_4bit=True --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 1 --device cuda:0
```

## Results
### Llama 2 7B
#### v0.4.0
hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 25, batch_size: 1
|    Tasks    |Version|Filter|n-shot| Metric |Value |   |Stderr|
|-------------|-------|------|-----:|--------|-----:|---|-----:|
|arc_challenge|Yaml   |none  |    25|acc     |0.5026|±  |0.0146|
|             |       |none  |    25|acc_norm|0.5358|±  |0.0146|
acc_norm: 53.58% vs 52.9% (hf) //
9 mins 45 secs (hypatia)

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 10, batch_size: 1
|  Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|---------|-------|------|-----:|--------|-----:|---|-----:|
|hellaswag|Yaml   |none  |    10|acc     |0.5927|±  |0.0049|
|         |       |none  |    10|acc_norm|0.7858|±  |0.0041|
acc_norm: 78.58% vs 78.55% (hf) //
1 hr 13 mins 10 secs (hypatia)

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
avg acc: 27.4563/57 = 0.4817% vs 48.32% (hf)
1 hr 27 mins 16 secs (hypatia)

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
|    Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|--------------|-------|------|-----:|------|-----:|---|-----:|
|truthfulqa_mc2|Yaml   |none  |     0|acc   |0.4531|±  |0.0156|
acc: 45.31% vs 45.57% (hf)
2 mins 44 secs (hypatia)

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1
|  Tasks   |Version|Filter|n-shot|Metric|Value |   |Stderr|
|----------|-------|------|-----:|------|-----:|---|-----:|
|winogrande|Yaml   |none  |     5|acc   |0.6638|±  |0.0133|
acc: 66.38% vs 71.74% (hf)
48 secs

hf (pretrained=meta-llama/Llama-2-7b-chat-hf), gen_kwargs: (), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|  Filter  |n-shot|  Metric   |Value |   |Stderr|
|-----|-------|----------|-----:|-----------|-----:|---|-----:|
|gsm8k|Yaml   |get-answer|     5|exact_match|0.2274|±  |0.0115|
exact_match: 22.74% vs 7.35% (hf - acc?)




#### HF version
hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 10, batch_size: 8
|  Task   |Version| Metric |Value |   |Stderr|
|---------|------:|--------|-----:|---|-----:|
|hellaswag|      0|acc     |0.5970|±  |0.0049|
|         |       |acc_norm|0.7849|±  |0.0041|

78.49% vs 78.55% (HF) //
2 hrs 57 mins (galileo)


hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 0, batch_size: 8
|    Task     |Version|Metric|Value |   |Stderr|
|-------------|------:|------|-----:|---|-----:|
|truthfulqa_mc|      1|mc1   |0.3023|±  |0.0161|
|             |       |mc2   |0.4531|±  |0.0156|

mc2 acc: 45.31% vs 45.57% (HF) //
22 seconds (galileo)


hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 8
|   Task   |Version|Metric|Value |   |Stderr|
|----------|------:|------|-----:|---|-----:|
|winogrande|      0|acc   |0.7269|±  |0.0125|

72.69% vs 71.74% (hf) //
1 min 56 seconds (galileo)


hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 4
|Task |Version|Metric|Value |   |Stderr|
|-----|------:|------|-----:|---|-----:|
|gsm8k|      0|acc   |0.1334|±  |0.0094|

13.34% vs 7.35% (hf) //
43 mins 25 seconds (galileo)

### Llama2 7B GPTQ
hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 0, batch_size: 1
|    Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|--------------|-------|------|-----:|------|-----:|---|-----:|
|truthfulqa_mc1|Yaml   |none  |     0|acc   |0.2901|±  |0.0159|
|truthfulqa_mc2|Yaml   |none  |     0|acc   |0.4412|±  |0.0156|

44.12% vs 49.95% (Chat AWQ) vs 39.32% (Non-Chat GPTQ) //
15 mins 25 seconds (brahe GPU:0)

### HuggingFace Open LLM Leaderboard
| Model                 | Average | ARC  | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K |
|-----------------------|---------|------|-----------|------|------------|------------|-------|
| meta-llama/llama-2-7b-chat-hf | 50.74    | 52.9 | 78.55     | 48.32 | 45.57       | 71.74      | 7.35  |
