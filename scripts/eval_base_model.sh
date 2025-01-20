MODEL_BASE=None
MODEL_NAME=llama3.1_8b_medical

if [[ $MODEL_NAME == *"llama3.2"* ]]; then 
    ckpt=Llama-3.2-3B-Instruct
elif [[ $MODEL_NAME == *"llama3.1_8b"* ]]; then
    ckpt=Meta-Llama-3.1-8B-Instruct-ysl
fi


MODEL_PATH=None
TASK_PATH=None



# bash scripts/eval/eval_models_per_dataset_base.sh $MODEL_BASE $MODEL_NAME greedy 1 1 $MODEL_PATH
bash scripts/eval/eval_models_per_dataset_base.sh $MODEL_BASE $MODEL_NAME sc 16 1 $MODEL_PATH