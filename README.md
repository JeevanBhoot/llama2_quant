## Setting Up Miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

```

## Installing EleutherAI LM Evaluation Harness
Same version that HuggingFace uses for Open LLM Leaderboard
```
conda create -n llama2_env python=3.10
conda activate llama2_env
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
pip install -e .
pip install gekko
conda install -c "nvidia/label/cuda-12.1.0" cuda
pip install -e ".[auto-gptq]"
```

## Llama2 Access
Request access at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
Log in with `huggingface-cli login` before running evaluation harness

## Evaluation Commands
```
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True --tasks hellaswag --device cuda:0 --num_fewshot 10 --batch_size 8

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True --tasks truthfulqa_mc --device cuda:0 --num_fewshot 0 --batch_size 8

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True --tasks winogrande --device cuda:0 --num_fewshot 5 --batch_size 8

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True --tasks gsm8k --device cuda:0 --num_fewshot 5 --batch_size 4
```

## Results
hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 10, batch_size: 8
|  Task   |Version| Metric |Value |   |Stderr|
|---------|------:|--------|-----:|---|-----:|
|hellaswag|      0|acc     |0.5970|±  |0.0049|
|         |       |acc_norm|0.7849|±  |0.0041|

78.48% vs 78.59%
2 hrs 57 mins

hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 0, batch_size: 8
|    Task     |Version|Metric|Value |   |Stderr|
|-------------|------:|------|-----:|---|-----:|
|truthfulqa_mc|      1|mc1   |0.3023|±  |0.0161|
|             |       |mc2   |0.4531|±  |0.0156|

37.77% average vs 38.76% (HF)
22 seconds

hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 8
|   Task   |Version|Metric|Value |   |Stderr|
|----------|------:|------|-----:|---|-----:|
|winogrande|      0|acc   |0.7269|±  |0.0125|

72.69% vs 74.03%
1 min 56 seconds

hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 4
|Task |Version|Metric|Value |   |Stderr|
|-----|------:|------|-----:|---|-----:|
|gsm8k|      0|acc   |0.1334|±  |0.0094|

13.34% vs 14.48%
43 mins 25 seconds