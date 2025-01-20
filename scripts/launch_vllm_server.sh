# MODEL=/mnt/hwfile/medai/LLMModels/Model/Qwen2-7B-Instruct-lys
# MODEL=/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3.1-8B-Instruct
MODEL=/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct

srun -p medai_llm --quotatype=auto --gres=gpu:1 vllm serve ${MODEL} \
    --port 10002 \
    --dtype auto \
    --guided-decoding-backend lm-format-enforcer