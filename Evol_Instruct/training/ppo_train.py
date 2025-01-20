# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil

import torch
from collections import defaultdict
from accelerate import PartialState
from Evol_Instruct import client
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from dataclasses import dataclass, field
from typing import Optional
from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from peft import PeftModel
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from Evol_Instruct.utils.utils import add_proxy
import torch.distributed as dist

def print_on_main(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

"""
python -i examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""


def obtain_dataset_each(data):
    new_data = []
    for item in data:
        new_data.append({
            "prompt": item['conversations'][0]['value']
        })

        
    dataset = Dataset.from_list(new_data)
    return dataset



def obtain_dataset(data_path, split_num=100):
    total_data = client.read(data_path)

    dataset = obtain_dataset_each(total_data)
    dataset = dataset.train_test_split(test_size=split_num)
    return dataset['train'], dataset['test']
    

@dataclass
class PPOScriptArguments:
    dataset_name: str = field(metadata={"help": "Name of the dataset to use."}, default=None)
    data_path: str = field(metadata={"help": "Path to the data file."}, default=None)
    test_num: int = field(metadata={"help": "Number of test samples."}, default=100)
    
    # if load a lora-tuned sft model
    sft_lora_path : list[str] = field(metadata={"nargs": "+", "help": "Path to the lora-tuned sft model."}, default_factory=list)
    reward_lora_path: list[str] = field(metadata={"nargs": "+", "help": "Path to the lora-tuned reward model."}, default_factory=list)
    
    
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False

if __name__ == "__main__":
    parser = HfArgumentParser((PPOScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    # shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>" if "3.1-8B" in model_args.model_name_or_path else tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    value_device = value_model.device
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_device = reward_model.device
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    policy_device = policy.device
    
    if script_args.sft_lora_path is not None:
        for path in script_args.sft_lora_path:
            policy = PeftModel.from_pretrained(policy, path).merge_and_unload()
            print_on_main(f"Loaded lora-tuned sft model from {path} into policy")
        policy = policy.to(policy_device)
        
    if script_args.reward_lora_path is not None:
        for path in script_args.sft_lora_path:
            reward_model = PeftModel.from_pretrained(reward_model, path).merge_and_unload()
            print_on_main(f"Loaded lora-tuned reward model from {path} into reward_model")
            
        for path in script_args.reward_lora_path:
            reward_model = PeftModel.from_pretrained(reward_model, path).merge_and_unload()
            print_on_main(f"Loaded lora-tuned reward model from {path} into reward_model")
        reward_model = reward_model.to(reward_device)   
        
        for path in script_args.sft_lora_path:
            value_model = PeftModel.from_pretrained(value_model, path).merge_and_unload()
            print_on_main(f"Loaded lora-tuned value model from {path} into value_model")
        for path in script_args.reward_lora_path:
            value_model = PeftModel.from_pretrained(value_model, path).merge_and_unload()
            print_on_main(f"Loaded lora-tuned value model from {path} into value_model")
        value_model = value_model.to(value_device)
            
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    # dataset = load_dataset(
    #     script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    # )
    # eval_samples = 100
    # train_dataset = dataset.select(range(len(dataset) - eval_samples))
    # eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    
    train_dataset, eval_dataset = obtain_dataset(script_args.data_path, split_num=script_args.test_num)
    
    dataset_text_field = "prompt"

    def prepare_dataset(dataset: Dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            messages = [[{"role": 'user', "content": each}] for each in element['prompt']]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = tokenizer(
                text,
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    
    
    with add_proxy():
        trainer = PPOTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
        trainer.train()

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)

        trainer.generate_completions()