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

# Evaluation Command
```
lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks hellaswag --device cuda:0 --batch_size 8
```