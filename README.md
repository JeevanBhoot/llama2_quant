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

Run the entire HuggingFace Open LLM Leaderboard evaluation suite with the following command:
```
bash eval.sh <harness_version> <gptq_flag> <output_filename>
```
`<harness_version>` is either `v0.4.0` or `b281b09`.

`<gptq_flaq>` is either `True` or `False`. If True, `pretrained=TheBloke/Llama-2-7B-Chat-GPTQ` weights will be used. 
Else, `meta-llama/Llama-2-7b-chat-hf` weights will be used.

The terminal output is saved to `<output_filename` inside `logs/`. If `<output_filename>` is not provided, 
a default filename will be given e.g. `output_gptq_v0.4.0_{timestamp}.log`

## HuggingFace Open LLM Leaderboard Settings
- ARC: 25-shot, arc_challenge (acc_norm)
- HellaSwag: 10-shot, hellaswag (acc_norm)
- TruthfulQA: 0-shot, truthfulqa_mc (mc2)
- MMLU: 5-shot, mmlu (v0.4.0) (average of all the results acc)
- Winogrande: 5-shot, winogrande (acc)
- GSM8k: 5-shot, gsm8k (acc)

## Results
| Model                 | Average | ARC  | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K |
|-----------------------|---------|------|-----------|------|------------|------------|-------|
| meta-llama/llama-2-7b-chat-hf | 50.74    | 52.9 | 78.55     | 48.32 | 45.57       | 71.74      | 7.35  |
| TheBloke/Llama-2-7B-GPTQ         | 48.48   | 52.05| 77.59     | 43.99| 39.32       | 72.93      | 5     |
| TheBloke/Llama-2-7b-Chat-AWQ     | 29.14   | 27.22| 25.48     | 24.67| 49.95       | 47.51      | 0     |


| Model                 |Harness Version| Average | ARC  | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K |
|-----------------------|----------|---------|------|-----------|------|------------|------------|-------|
| meta-llama/llama-2-7b-chat-hf | v0.4.0   | 52.31 | 53.58 | 78.58  | 47.24* | 45.31  | 66.38  | 22.74** |
| meta-llama/llama-2-7b-chat-hf | b281b09  | 52.67 | 52.30 | 78.52  | 48.17% | 45.31  | 73.01  | 18.73   |
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
