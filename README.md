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
hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 10, batch_size: 8
|  Task   |Version| Metric |Value |   |Stderr|
|---------|------:|--------|-----:|---|-----:|
|hellaswag|      0|acc     |0.5970|±  |0.0049|
|         |       |acc_norm|0.7849|±  |0.0041|

78.48% vs 78.59% (HF) //
2 hrs 57 mins (galileo)


hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 0, batch_size: 8
|    Task     |Version|Metric|Value |   |Stderr|
|-------------|------:|------|-----:|---|-----:|
|truthfulqa_mc|      1|mc1   |0.3023|±  |0.0161|
|             |       |mc2   |0.4531|±  |0.0156|

37.77% average vs 38.76% (HF) //
22 seconds (galileo)


hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 8
|   Task   |Version|Metric|Value |   |Stderr|
|----------|------:|------|-----:|---|-----:|
|winogrande|      0|acc   |0.7269|±  |0.0125|

72.69% vs 74.03% //
1 min 56 seconds (galileo)


hf-causal-experimental (pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 4
|Task |Version|Metric|Value |   |Stderr|
|-----|------:|------|-----:|---|-----:|
|gsm8k|      0|acc   |0.1334|±  |0.0094|

13.34% vs 14.48% //
43 mins 25 seconds (galileo)

### Llama2 7B GPTQ
hf (pretrained=TheBloke/Llama-2-7B-Chat-GPTQ,gptq=True,load_in_4bit=True), gen_kwargs: (), limit: None, num_fewshot: 0, batch_size: 1
|    Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|--------------|-------|------|-----:|------|-----:|---|-----:|
|truthfulqa_mc1|Yaml   |none  |     0|acc   |0.2901|±  |0.0159|
|truthfulqa_mc2|Yaml   |none  |     0|acc   |0.4412|±  |0.0156|

44.12% vs 49.95% (Chat AWQ) vs 39.32% (Non-Chat GPTQ) //
15 mins 25 seconds (brahe GPU:0)