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


from itertools import chain

import torch.nn as nn


import warnings
from peft import PeftModel
from datasets import DatasetDict
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset, features
from typing import List
import numpy as np
from transformers import TrainerCallback
import pathlib
import torch
from accelerate import PartialState
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, HfArgumentParser
import random
from Evol_Instruct.MCTS.tree_node import MedMCTSNode
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
from Evol_Instruct.models.modeling_value_llama import ProcessRewardModel, ProcessTrainer, ProcessDataCollator
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
    use_hrm: bool = False
    look_back_factor: float = 0
    use_soft_look_back: bool = False 
    
    look_back_head: bool = False
    
    
    warmup_training: bool = False 
    warmup_training_steps: int = 100
    
    score_lr: float = None
    
    loss_type: str = 'huber'

def tokenize_fn(features, tokenizer, step_separator, max_length, max_completion_length, train_on_last_step_only, is_eval):
    # now we only train on last step, we just need to set the input_ids's last token to corresponding label 

    # if train_on_last_step_only and not is_eval:
    #     labels = [-100] * (len(features["labels"]) - 1) + [int(features["labels"][-1])]
    # else:
    #     labels = [int(label) if isinstance(label, bool) else label for label in features['labels']]
    # response = features['completions']
    completions = features['completions']
    
    assert len(completions) == len(features['labels']), f"Expect to see {len(completions)} labels for steps, got {len(features['labels'])}"
    completion_ids = [
        tokenizer(completion + "\n\n", add_special_tokens=False)['input_ids'] for completion in completions
    ]
    messages = [
        {"role": "user", "content": features['prompt']},
        # {"role": "assistant", "content": response}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response_id = list(chain(*completion_ids))
    pre_response_id = tokenizer(prompt_text, add_special_tokens=False)['input_ids']

    input_ids = pre_response_id + response_id
    completion_index = []
    
    labels = [-100] * len(input_ids)
    
    # labels[-1] = features["labels"][-1]
    penalties = [-100] * len(input_ids)
    # penalties[-1] = int(features['penalties'][-1] > 0)
    
    
    for i, completion in enumerate(completion_ids):
        if i == 0:
            completion_index.append(len(completion) + len(pre_response_id) - 1)
        else:
            completion_index.append(completion_index[-1] + len(completion))
        if i > 0:
            labels[completion_index[-1]] = features["labels"][i]
            
            penalties[completion_index[-1]] = int(features['penalties'][i] > 0)
    
    
    assert len(input_ids) == len(labels)
    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        penalties = penalties[:max_length]
    
    # labels = labels + penalties
    
    return {"input_ids": input_ids, "labels": labels, "penalties": penalties}

def format_one_item(path, prompt, script_args, use_soft_training=False,):
    # use_soft_training = True
    positive_thr = script_args.positive_thr
    look_back_factor = script_args.look_back_factor
    use_soft_look_back = script_args.use_soft_look_back
    look_back_head = script_args.look_back_head
    use_hrm = script_args.use_hrm
    # completions = "\n\n".join([f"Step {i}: {each_step[0]}" for i, each_step in enumerate(path)])
    completions = []
    step_idx = 0
    for i, each_step in enumerate(path):
        splitted_step = each_step[0].split("\n\nStep")

        completions.append(f"Step {step_idx}: {each_step[0]}")
        step_idx += len(splitted_step)
            
    # completions = [f"Step {i}: {each_step[0]}" for i, each_step in enumerate(path)]
    label = [each_step[1] for i, each_step in enumerate(path)]
    penalty_list = [0 for i, value in enumerate(label)]
    if look_back_factor > 0 and (not look_back_head):
        
        for i, value in enumerate(label):
            if i <= 1:
                continue 
            if use_soft_look_back:
                if i == len(label) - 1:
                    penalty = 0
                else:
                    if value < label[i-1] and value < label[i+1]:
                        penalty = look_back_factor * (label[i-1] - label[i+1])
                    else:
                        penalty = look_back_factor * (label[i-1] - value)
                    # penalty = look_back_factor * (label[i-1] - value)
                # penalty = look_back_factor * 
            else:
                penalty = look_back_factor * (-value + label[i-1])
            penalty_list[i] = penalty
            label[i] = value - penalty
            
    if not use_soft_training:
        if use_hrm:
            label = [1.0 if (value > positive_thr and (i > 0 and value >= label[i-1])) else 0 for i, value in enumerate(label)]
        else:
            label = [1.0 if value > positive_thr else 0.0 for value in label]
    return {'prompt': prompt, 'completions': completions, 'labels': label, "penalties": penalty_list}


def path_uncertainty_score(path):
    # values = [node.value for node in path]
    values = [reason_step[1] for reason_step in path]
    # 使用方差衡量路径中 value 的波动程度
    return np.var(values)

def collect_all_complete_paths(root: MedMCTSNode) -> List[List[MedMCTSNode]]:
    """收集所有从根节点到叶子节点的完整路径"""
    paths = []

    def dfs(node: MedMCTSNode, current_path: List[MedMCTSNode]):
        current_path.append(node)
        if not node.children or node.is_completed:
            if node.value in (0, 1):  # 确保是叶子节点且有明确结果
                paths.append(current_path.copy())
        else:
            for child in node.children:
                dfs(child, current_path)
        current_path.pop()

    dfs(root, [])
    return paths



def obtain_dataset_each(data, script_args, tokenizer, filter_invalid=False, use_soft_training=False, max_length=None,):
    train_pair_per_instance = script_args.train_pair_per_instance
    new_data = []
    for item in tqdm(data, total=len(data), desc='pre-processing data'):
            # shuffle_list()
        # train pair per instance is a deprecated param
        if filter_invalid:
            if len(item['neg']) == 0 or len(item['pos']) == 0:
                
                continue 
        
        def process_cat_path(item, cat, train_pair_per_instance):
            all_paths = [trajectories[0] for trajectories in item[cat]] 
            if train_pair_per_instance == 0:
                return [(1, path) for path in all_paths]
            scored_paths = [(path_uncertainty_score(path), path) for path in all_paths]
            scored_paths.sort(reverse=True, key=lambda x: x[0])
            return scored_paths

        pos_uncertain_score = process_cat_path(item, 'pos', train_pair_per_instance) if len(item['pos']) > 0 else []
        neg_uncertain_score = process_cat_path(item, 'neg', train_pair_per_instance) if len(item['neg']) > 0 else []
        if train_pair_per_instance == -1:
            # each select min(len(pos), len(neg))
            train_pair_per_instance = min(len(item['pos']), len(item['neg']))
        elif train_pair_per_instance == 0:
            train_pair_per_instance = max(len(item['pos']), len(item['neg']))
        sampled_pos_paths = [p for s, p in pos_uncertain_score[:train_pair_per_instance]]
        sampled_neg_paths = [p for s, p in neg_uncertain_score[:train_pair_per_instance]]
        
        problem = item['question']
        
        scored_paths = sampled_pos_paths + sampled_neg_paths 
        
        for path in scored_paths:
            new_data.append(
                format_one_item(path, problem, script_args, use_soft_training=use_soft_training, )
            )


        
    # shuffle_list(lst=new_data)
    dataset = Dataset.from_list(new_data)
    # filter overlong sequences 
    # {'prompt': prompt, 'completions': completions, 'labels': label, "penalties": penalty_list}
    dataset = dataset.filter(
        lambda x: len(tokenizer.apply_chat_template([{"role": "user", "content": x['prompt']},{"role": "assistant", "content": x['completions']}], tokenize=True)) <= max_length,
        # input_columns=["prompt", "completions"],
        num_proc=16,
    )
    return dataset 



def obtain_dataset(tokenizer, script_args, args, split_ratio=0.1,):
    data_path = script_args.data_path
    filter_invalid = script_args.filter_invalid
    use_soft_training = script_args.use_soft_training
    use_rollout_label = script_args.use_rollout_label
    train_pair_per_instance = script_args.train_pair_per_instance
    total_data = client.read(data_path)

    # random.shuffle(total_data)
    total_idx = list(range(len(total_data)))
    # Dataset.select()
    
    test_num = int(len(total_data) * split_ratio)
    print_on_main(f"Test num: {test_num}", flush=True)
    test_idx = set(uniform_sample(total_idx, test_num))
    print_on_main("Obtain test idx", flush=True)
    train_data = []
    test_data = []
    for i in total_idx:
        if i in test_idx: 
            test_data.append(total_data[i])
        else:
            train_data.append(total_data[i])
            
    print_on_main(f"Splitting train/test sets, train: {len(train_data)}, test: {len(test_data)}", flush=True)
    # train_data = total_data[:train_num]
    # test_data = total_data[train_num:]
    train_dataset = obtain_dataset_each(train_data, script_args, tokenizer, filter_invalid, use_soft_training=use_soft_training, max_length=args.max_length,  )
    eval_dataset = obtain_dataset_each(test_data, script_args, filter_invalid=False, max_length=args.max_length, use_soft_training=False, tokenizer=tokenizer)
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
                            # "labels": features.Sequence(features.Value("float32")),
                            "input_ids": features.Sequence(features.Value("int64")),
                            "penalties": features.Sequence(features.Value("int64"))
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
                    "penalties": features.Sequence(features.Value("int64"))
                }
            ),
        )
    
    return {"train": train_dataset, "test": eval_dataset}
    
    
def soft_label_compute_func(outputs, labels, num_items_in_batch):
    # labels is a [B, N] tensor, should be converted to [B, N, 2] tensor
    # logits is a [B, N, 2] tensor
    logits = outputs['logits']
    # look_back_penalties = labels[:, logits.shape[1]:]
    # labels = labels[:, :logits.shape[1]]
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



class FreezeCallback(TrainerCallback):
    def __init__(self, freeze_steps=100):
        self.freeze_steps = freeze_steps

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step < self.freeze_steps:
            # 冻结除 score 外的所有层
            for name, param in model.named_parameters():
                if "score" not in name:
                    param.requires_grad = False
        else:
            # 解冻所有层
            is_lora_model = False
            for name, param in model.named_parameters():
                if "lora_" in name:
                    is_lora_model = True
                    break
                # param.requires_grad = True
            for name, param in model.named_parameters():
                if "lora_" in name or "score" in name:
                    param.requires_grad = True
                else:
                    if is_lora_model or "embed_tokens" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        return control



class CustomPRMTrainer(PRMTrainer):
    def __init__(self, *args,  loss_type="huber", epsilon=0.1, delta=0.5, score_lr=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.epsilon = epsilon
        # self.use_soft_training = use_soft_training
        self.delta = delta
        self.score_lr = score_lr
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
        # self.bce = nn.BCELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss()
    
    def create_optimizer(self):
        if self.score_lr == 0.0:
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = PRMTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            # score_param = [p for n, p in opt_model.named_parameters() if "score" in n]
            # other_param = [p for n, p in opt_model.named_parameters() if "score" not in n and p.requires_grad]
            
            lr = optimizer_kwargs["lr"]

            decay_parameters = self.get_decay_parameter_names(opt_model)

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "score" not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "score" not in n)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "score" in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.score_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "score" in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.score_lr,
                },
            ]
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        return self.optimizer
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            labels=inputs.get("labels", None),  # 有些 loss 需要 labels
        )
        logits = outputs.logits  # shape: (batch_size, seq_len, 2)

        # 获取 mask：哪些位置是要计算 loss 的（非 -100）
        labels = inputs["labels"]
        mask = labels != -100  # shape: (batch_size, seq_len)

        # 提取有效的预测和标签
        valid_logits = logits[mask]
        label_dtype = labels.dtype
        valid_labels = labels[mask].to(valid_logits.dtype)  # shape: (num_valid, )
        valid_class1_logits = valid_logits[:, 1]  # shape: (num_valid, )

        # 对 label 做 smoothing（适用于 BCE）
        if self.loss_type == "bce" or self.loss_type == "hybrid":
            # smoothed_labels = valid_labels * (1 - self.epsilon) + 0.5 * self.epsilon
            smoothed_labels = torch.clamp(valid_labels, min=0, max=1 - 1e-4)
        else:
            smoothed_labels = valid_labels
        # if label_dtype == torch.int64:
        #     # evaluation loop
        #     self.loss_type = "bce"
        # smoothed_labels = valid_labels 
        
        # if not self.use_soft_training:
        #     cls_labels = torch.where(valid_labels > 0, torch.ones_like(valid_labels), torch.zeros_like(valid_labels))
        # else:
        #     cls_labels = smoothed_labels
        # 根据 loss_type 计算 loss
        if self.loss_type == "mse":
            preds = torch.softmax(valid_logits, dim=-1)[:, 1]
            loss = nn.functional.mse_loss(preds, smoothed_labels)
        elif self.loss_type == "huber":
            preds = torch.softmax(valid_logits, dim=-1)[:, 1]
            loss = self.huber(preds, smoothed_labels).mean()
        elif self.loss_type == "bce":
            loss = self.bce(valid_class1_logits, smoothed_labels).mean()
        elif self.loss_type == "hybrid":
            bce_loss = self.bce(valid_class1_logits, smoothed_labels)
            huber_loss = self.huber(torch.sigmoid(valid_class1_logits), smoothed_labels)
            loss = 0.5 * bce_loss + 0.5 * huber_loss
            loss = loss.mean()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


def param_groups_func(model, base_lr, score_lr):
    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'score' in name:
            groups.append({'params': [param], 'lr': score_lr})
        else:
            groups.append({'params': [param], 'lr': base_lr})
    return groups


def param_groups_func(model, base_lr):
    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            groups.append({'params': [param], 'lr': base_lr * 0.1})
        else:
            groups.append({'params': [param], 'lr': base_lr})
    return groups

    

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
    if script_args.look_back_head:
        model = ProcessRewardModel.from_pretrained(
            model_config.model_name_or_path, num_labels=2, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
        trainer_cls = ProcessTrainer
        collate_fn = collator = ProcessDataCollator(tokenizer)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_config.model_name_or_path, num_labels=2, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
        # trainer_cls = PRMTrainer
        trainer_cls = CustomPRMTrainer
        collate_fn = None
        
    if script_args.tuned_lora_path is not None and script_args.tuned_lora_path != ['None']:
        for lora_path in script_args.tuned_lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print_on_main(f"Load lora weights from {lora_path}")
    for n, p in model.named_parameters():
        if "embed_tokens" in n:
            p.requires_grad = False 
        else:
            p.requires_grad = True
    # print_on_main(model)
    print_on_main(script_args)
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
                                training_args, split_ratio=script_args.test_split_ratio, )
    
    
    with add_proxy():
        trainer = trainer_cls(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            peft_config=get_peft_config(model_config),
            data_collator=collate_fn,
            # compute_loss_func=soft_label_compute_func if script_args.use_soft_training else None,
            callbacks=[FreezeCallback(script_args.warmup_training_steps)] if script_args.warmup_training else None,
            # use_soft_training=script_args.use_soft_training,
            loss_type=script_args.loss_type,
        )
        # results = trainer.evaluate()
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
