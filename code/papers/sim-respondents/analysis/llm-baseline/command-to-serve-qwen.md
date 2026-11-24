use global venv

transformers serve Qwen/Qwen3.5-0.8B --port 8000 --continuous-batching --dtype bfloat16


vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --language-model-only


vllm serve Qwen/Qwen3.5-4B \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --language-model-only

vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --language-model-only


to get preds:

python code/papers/sim-respondents/analysis/llm-baseline/predict_qwen.py \
  --model Qwen/Qwen3.5-9B \
  --personas respondent \
  --thinking off on
