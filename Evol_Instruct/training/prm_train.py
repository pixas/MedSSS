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

"""
Full training:
python examples/scripts/prm.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/prm800k \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50

LoRA:
python examples/scripts/prm.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/prm800k \
    --output_dir Qwen2-0.5B-Reward-LoRA \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

from itertools import chain
import warnings
from peft import PeftModel
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset, features
import pathlib
import torch
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, HfArgumentParser
import random

from trl import (
    ModelConfig,
    PRMConfig,
    PRMTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from Evol_Instruct import client
from Evol_Instruct.utils.utils import add_proxy
from Evol_Instruct.training.value_train import uniform_sample, shuffle_list, print_on_main

@dataclass
class PRMScriptArguments:
    dataset_name: str = field(metadata={"help": "Name of the dataset to use."}, default=None)
    data_path: str = field(metadata={"help": "Path to the data file."}, default=None)
    test_split_ratio: float = field(metadata={"help": "Ratio of the test split."}, default=0.1)
    
    # if load a lora-tuned sft model
    tuned_lora_path: list[str] = field(metadata={"nargs": "+", "help": "Path to the lora-tuned model."}, default_factory=list)
    train_pair_per_instance: int = field(metadata={"help": "Number of training pairs per instance."}, default=1)
    
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False
    
    use_soft_training: bool = False
    use_rollout_label: bool = False
    filter_invalid: bool = False
    positive_thr: float = 0.5

def tokenize_fn(features, tokenizer, step_separator, max_length, max_completion_length, train_on_last_step_only, is_eval):
    if train_on_last_step_only and not is_eval:
        labels = [-100] * (len(features["labels"]) - 1) + [int(features["labels"][-1])]
    else:
        labels = [int(label) if isinstance(label, bool) else label for label in features['labels']]
    response = step_separator.join(features['completions']) 
    # separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
    
    messages = [
        {"role": "user", "content": features['prompt']},
        {"role": "assistant", "content": response}
    ]
    # prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    # prompt_ids = tokenizer(prompt_text)['input_ids']
    completion_ids = [
        tokenizer(completion + step_separator, add_special_tokens=False)['input_ids'] for completion in features['completions']
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    response_begin_index = input_text.index(response)
    pre_response_input = input_text[:response_begin_index]
    after_response_input = input_text[response_begin_index + len(response):]
    
    # response_id = tokenizer(response)['input_ids']
    response_id = list(chain(*completion_ids))
    pre_response_id = tokenizer(pre_response_input, add_special_tokens=False)['input_ids']
    after_response_id = tokenizer(after_response_input, add_special_tokens=False)['input_ids']
    
    input_ids = pre_response_id + response_id + after_response_id
    
    # completion_labels_list = list(chain(*completion_labels)) 
    completion_labels = [[-100] * (len(completion) - 1) + ([label] if i > 0 and label is not None else [-100]) for i, (completion, label) in enumerate(zip(completion_ids, labels))]
    labels = [-100] * len(pre_response_id) + list(chain(*completion_labels)) + [-100] * len(after_response_id)
    
    
    
    # input_ids = tokenizer(input_text)['input_ids']
    
    
    # labels = [-100] * len(prompt_ids) + completion_labels_list + [-100]
    
    # if max_completion_length is not None:
    #     completion_ids = completion_ids[:max_completion_length]
    #     labels = labels[:max_completion_length]

    # if tokenizer.bos_token_id is not None:
    #     prompt_ids = [tokenizer.bos_token_id] + prompt_ids

    # input_ids = prompt_ids + completion_ids
    # labels = [-100] * len(prompt_ids) + labels
    
    assert len(input_ids) == len(labels)
    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    
    return {"input_ids": input_ids, "labels": labels}

def format_one_item(item, question, use_soft_training=False, positive_thr=0.5):
    completion = [f"Step {m}: " + step['step'].rstrip('.') + '.' for m, step in enumerate(item)]
    prompt = question
    label = [(True if item[m]['value'] > positive_thr else False) if not use_soft_training else item[m]['value'] for m in range(len(item))]
    return {'prompt': prompt, 'completions': completion, 'labels': label}

def obtain_dataset_each(data, filter_invalid=False, use_soft_training=False, use_rollout_label=False, train_pair_per_instance = -1, positive_thr=0.5):

    new_data = []
    for item in data:
            # shuffle_list()
        if filter_invalid:
            if len(item['pos']) == 0 or len(item['neg']) == 0:
                # only select one item 
                category = 'pos' if len(item['pos']) > 0 else 'neg'
                shuffle_list(item[category])
                new_data.append(format_one_item(item[category][0], item['question'], use_soft_training, positive_thr))
                continue 
        for category in ['pos', 'neg']:
            if train_pair_per_instance > 0:
                shuffle_list(item[category])
                # item[category].sort(key=lambda x: len(x[0]), reverse=True)
                # if len(item[category]) >= train_pair_per_instance:
                    
                #     item[category][0], item[category][train_pair_per_instance - 1] = item[category][train_pair_per_instance - 1], item[category][0]
                cut_length = train_pair_per_instance
            else:
                cut_length = len(item[category])
            for i in range(min(cut_length, len(item[category]))):
                new_data.append(
                    format_one_item(item[category][i], item['question'], use_soft_training, positive_thr)
                )

        
    shuffle_list(new_data)
    dataset = Dataset.from_list(new_data)
    
    return dataset 

def obtain_dataset(tokenizer, script_args, args, split_ratio=0.1, train_pair_per_instance=-1):
    data_path = script_args.data_path
    filter_invalid = script_args.filter_invalid
    use_soft_training = script_args.use_soft_training
    use_rollout_label = script_args.use_rollout_label
    
    total_data = client.read(data_path)
    # random.shuffle(total_data)
    total_idx = list(range(len(total_data)))
    # Dataset.select()
    
    test_num = int(len(total_data) * split_ratio)
    test_idx = set(uniform_sample(total_idx, test_num))
    train_data = []
    test_data = []
    for i in total_idx:
        if i in test_idx: 
            test_data.append(total_data[i])
        else:
            train_data.append(total_data[i])
            

    # train_data = total_data[:train_num]
    # test_data = total_data[train_num:]
    train_dataset = obtain_dataset_each(train_data, filter_invalid, use_soft_training, use_rollout_label=use_rollout_label, train_pair_per_instance=train_pair_per_instance, positive_thr=script_args.positive_thr)
    eval_dataset = obtain_dataset_each(test_data, filter_invalid=False, use_rollout_label=use_rollout_label, use_soft_training=False, positive_thr=script_args.positive_thr)
    fn_kwargs = {
                    "tokenizer": tokenizer,
                    "step_separator": args.step_separator,
                    "max_length": args.max_length,
                    "max_completion_length": args.max_completion_length,
                    "train_on_last_step_only": args.train_on_last_step_only,
                }
    train_fn_kwargs = {**fn_kwargs, "is_eval": False}
    train_dataset = train_dataset.map(
        tokenize_fn,
        fn_kwargs=train_fn_kwargs,
        num_proc=args.dataset_num_proc,
        remove_columns=train_dataset.features,
        desc='Tokenizing train dataset',
        features=features.Features(  # needed to avoid map to cast labels to bool
                        {
                            "labels": features.Sequence(features.Value("int64")) if not use_soft_training else features.Sequence(features.Value("float32")),
                            "input_ids": features.Sequence(features.Value("int64")),
                        }
                    ),
    )
    eval_fn_kwargs = {**fn_kwargs, "is_eval": True}
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize_fn,
            fn_kwargs=eval_fn_kwargs,
            num_proc=args.dataset_num_proc,
            remove_columns=eval_dataset.features,
            desc="Tokenizing eval dataset",
            features=features.Features(  # needed to avoid map to cast labels to bool
                {
                    "labels": features.Sequence(features.Value("int64")),
                    "input_ids": features.Sequence(features.Value("int64")),
                }
            ),
        )
    
    return {"train": train_dataset, "test": eval_dataset}
    
    
def soft_label_compute_func(outputs, labels, num_items_in_batch):
    # labels is a [B, N] tensor, should be converted to [B, N, 2] tensor
    # logits is a [B, N, 2] tensor
    logits = outputs['logits']
    # print_on_main(f"Soft label compute func: {logits.shape}, {labels.shape}")
    soft_labels = torch.zeros(labels.size(0), labels.size(1), 2, device=labels.device).to(logits.dtype)
    valid_mask = labels != -100
    indices = torch.nonzero(valid_mask, as_tuple=True)  # as_tuple=True 使得它返回行和列索引
    
    valid_probs = labels[valid_mask].to(soft_labels.dtype)
    soft_labels[indices[0], indices[1], 1] = valid_probs
    soft_labels[indices[0], indices[1], 0] = 1 - valid_probs
    ignore_mask = ~valid_mask
    soft_labels[ignore_mask] = 0
    
    loss = torch.sum(-soft_labels * torch.log_softmax(logits, dim=-1), dim=-1)
    
    # not compute the loss for the ignore_mask
    return torch.sum(loss * valid_mask) / (valid_mask).sum()


if __name__ == "__main__":
    parser = HfArgumentParser((PRMScriptArguments, PRMConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.step_separator = "\n\n"
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>" if "3.1-8B" in model_config.model_name_or_path else tokenizer.eos_token
        
    model = AutoModelForTokenClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=2, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    if script_args.tuned_lora_path is not None and script_args.tuned_lora_path != ['None']:
        for lora_path in script_args.tuned_lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print_on_main(f"Load lora weights from {lora_path}")
    
    
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_config.use_peft and model_config.lora_task_type != "TOKEN_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `TOKEN_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type TOKEN_CLS when using this script with PEFT.",
            UserWarning,
        )

    ##############
    # Load dataset
    ##############
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = obtain_dataset(tokenizer, 
                             script_args,
                             training_args, split_ratio=script_args.test_split_ratio, train_pair_per_instance=script_args.train_pair_per_instance)
    # dataset = dataset.filter(lambda x: len(x["completions"]) > 0)

    ##########
    # Training
    ##########
    with add_proxy():
        trainer = PRMTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            peft_config=get_peft_config(model_config),
            compute_loss_func=soft_label_compute_func if script_args.use_soft_training else None,
        )
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        ############################
        # Save model and push to Hub
        ############################
        trainer.save_model(training_args.output_dir)
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)