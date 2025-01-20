# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import defaultdict
from gettext import find
import os
import copy
from dataclasses import dataclass, field
import json
from datasets import Dataset
import logging
import pathlib
from sklearn.metrics import mean_squared_error
from transformers import DataCollatorWithPadding
from pathlib import Path
from typing import Dict, Optional, Sequence, List
import numpy as np
from transformers import  AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
from Evol_Instruct.models.modeling_value_llama import LlamaForValueFunction, obtain_value_cls
# from Evol_Instruct.training.sft_train import get_peft_state_non_lora_maybe_zero_3
# from ming.model.modeling_phi import PhiForCausalLM
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
import torch
import warnings
import transformers
from transformers import Trainer
from Evol_Instruct.training.value_trainer import ValueTrainer
import random

from transformers.trainer_pt_utils import LabelSmoother

# from Evol_Instruct.utils.conversations import get_default_conv_template, SeparatorStyle
import pdb

import warnings


from Evol_Instruct import client
from peft import LoraConfig, get_peft_model, PeftModel, LoftQConfig

import peft

from Evol_Instruct.utils.utils import add_proxy 

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    freeze_backbone: Optional[bool] = field(default=False)
    
    # mix of lora arguments
   
    
    wrap_modules: Optional[List[str]] = field(default_factory=list)
    
    # progressive params
    lora_name_or_path: Optional[str] = field(default=None)
    previous_lora_path: Optional[List[str]] = field(default_factory=list, metadata={
        "help": "please use ' ' to split each lora path to be loaded with their training order",
        "nargs": "+"
    })
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    test_size: float = field(default=0.1)
    prompt_type: str = field(default="llama",
                           metadata={"help": "prompt type"})
    is_base: bool = field(default=False, metadata={"help": "whether to use no-chat tuned model as the seed model"})
    even_value: bool = field(default=False, metadata={"help": "whether to use even the training set for the value model"})
    even_pair: bool = field(default=False, metadata={"help": "whether to use even the training set for the value model"})
    learn_rollout_value: bool = field(default=False, metadata={"help": "whether to learn the rollout value"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: Optional[str] = field(default=None, metadata={
        "help": "please use q,k,v,up,down,gate as the format where the lora module should wrap. Modules should be separated by commas"
    })

    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_use_rs: bool = False
    wrap_ffn: bool = True
    wrap_attn: bool = True
    only_head: bool = False
    
    cut_ratio: float = 1
    reduction: str = 'mean'
    score_lr: float = field(default=None, metadata={"help": "learning rate for the score linear layer"})
    # our method
    learn_by_stage: bool = field(default=False, metadata={"help": "whether to only train head in the first epoch"})
    

def compute_metrics(p):
    predictions, labels = p 
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}

def find_all_linear_names(model, ):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'switch']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'score' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('score')
    return list(lora_module_names)

def obtain_dataset(data_path, tokenizer, even_value=False, even_pair=False, learn_rollout_value=False):
    data = client.read(data_path)
    new_data = []
    for item in data:
        cur_item_data = []
        for category in ['pos', 'neg', 'inter']:
            for term in item[category]:
                if isinstance(term[0], str):
                    response = term[0]
                else:
                    response = "\n\n".join([f"Step {i}: " + step[0] for i, step in enumerate(term[0])])
                cur_item_data.append({
                    "id": item['id'],
                    "question": item['question'],
                    "response": response,
                    "label": term[2] if learn_rollout_value else term[1],
                    "inter": True if category == 'inter' else False
                })
        # sort the data in descending order by the label value 
        # remaining items with different labels
        cur_item_data.sort(key=lambda x: x['label'], reverse=True)
        # cur_item_data
        for j in range(1, len(cur_item_data)):
            if cur_item_data[j]['label'] < cur_item_data[j - 1]['label']:
                # new_data.extend(cur_item_data[j:])
                new_data.append(
                    {
                        "chosen": [{"role": "user", "content": cur_item_data[j-1]['question']}, {"role": "assistant", "content": cur_item_data[j-1]['response']}],
                        "rejected": [{"role": "user", "content": cur_item_data[j]['question']}, {"role": "assistant", "content": cur_item_data[j]['response']}],
                    }
                )
    dataset = Dataset.from_list(new_data)
    return dataset
    pass 

def obtain_dataset(data_path, tokenizer, even_value=False, even_pair=False, learn_rollout_value=False):
    data = client.read(data_path)
    new_data = []
    if even_pair:
        pos_neg_both_data = [item for item in data if item['pos'] and item['neg']]
        only_pos_data = [item for item in data if item['pos'] and (not item['neg'])]
        only_neg_data = [item for item in data if item['neg'] and (not item['pos'])]
        # obtain minimum
        min_number = min(len(pos_neg_both_data), len(only_pos_data), len(only_neg_data))
        temp_data = random.sample(pos_neg_both_data, min_number) + random.sample(only_pos_data, min_number) + random.sample(only_neg_data, min_number)
        # temp_data = pos_neg_both_data
        for item in temp_data:
            for category in ['pos', 'neg', 'inter']:
                for term in item[category]:
                    if isinstance(term[0], str):
                        response = term[0]
                    else:
                        response = "\n\n".join([f"Step {i}: " + step[0] for i, step in enumerate(term[0])])
                    new_data.append({
                        "id": item['id'],
                        "question": item['question'],
                        "response": response,
                        "label": term[2] if learn_rollout_value else term[1],
                        "inter": True if category == 'inter' else False
                    })
        pass
    else:
        for item in data:
            for category in ['pos', 'neg', 'inter']:
                for term in item[category]:
                    if isinstance(term[0], str):
                        response = term[0]
                    else:
                        response = "\n\n".join([f"Step {i}: " + step[0] for i, step in enumerate(term[0])])
                    new_data.append({
                        "id": item['id'],
                        "question": item['question'],
                        "response": response,
                        "label": term[2] if learn_rollout_value else term[1],
                        "inter": True if category == 'inter' else False
                    })
    
            
    if even_value:
        # count step distribution
        step_one_distribution = defaultdict(list)
        step_zero_distribution = defaultdict(list)
        evened_data = []
        for item in new_data:
            step = len(item['response'].split("\n\nStep"))
            if item['label'] < 1e-6:
                # step_zero_distribution[item['label']] += 1
                step_zero_distribution[step].append(item)   
            elif item['label'] > 1 - 1e-6:
                step_one_distribution[step].append(item)
        
        for key in step_one_distribution.keys():
            one_list = step_one_distribution[key]
            zero_list = step_zero_distribution[key]
            one_value_count = len(one_list)
            zero_value_count = len(zero_list)
            minimum = min(one_value_count, zero_value_count)
            if minimum  > 0:
                evened_data.extend(random.sample(one_list, minimum) + random.sample(zero_list, minimum))
        evened_data += [item for item in new_data if 1e-6 <= item['label'] <= 1 - 1e-6]
        # evened_data += [item for item in new_data if item['inter']]
            
        new_data = evened_data
        # count 0 and 1's items
        # zero_value_items = [item for item in data if item['label'] < 1e-6]
        # random.shuffle(zero_value_items)
        # one_value_items = [item for item in data if item['label'] > 1 - 1e-6]
        # random.shuffile(one_value_items)
        # other_value_items = [item for item in data if 1e-6 <= item['label'] <= 1 - 1e-6]
        # maintain_number = min(len(zero_value_items), len(one_value_items))
        # data = zero_value_items[:maintain_number] + one_value_items[:maintain_number] + other_value_items
        rank0_print("Load dataset with evening the value label")

    dataset = Dataset.from_list(new_data)
    original_columns = dataset.column_names 
    def preprocess_function(examples):
        output = {}
        # print(examples['output'][:2])
        text = [tokenizer.apply_chat_template([
            {"role": "user", "content": question},
            {"role": "assistant", "content": output}], tokenize=False,                                                 add_generation_prompt=False) for (question, output) in zip(examples['question'], examples['response'])]
        # print(text[:2])
        tokenized_text = [tokenizer(x, padding=True,) for x in text]    
        input_ids = [x['input_ids'] for x in tokenized_text]
        attention_mask = [x['attention_mask'] for x in tokenized_text]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": examples['label']
        }
        return output 
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=original_columns)
    return dataset

def split_dataset(dataset: Dataset, test_size=0.1):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split



if __name__ == "__main__":
    

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    rank0_print(training_args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
        


    if model_args.previous_lora_path != ['None']:
        # policy_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
        # policy_model = PeftModel.from_pretrained(policy_model, model_args.previous_lora_path).merge_and_unload().to(torch.bfloat16)
        # rank0_print("Loading LoRA weights from", model_args.previous_lora_path, "for policy model initialization")
        # policy_state_dict = policy_model.state_dict()
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        value_cls = obtain_value_cls(config)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=1,
            trust_remote_code=True,
            pad_token_id=tokenizer.pad_token_id,
            # reduction=training_args.reduction,
            # cut_ratio=training_args.cut_ratio
        )
        for lora_path in model_args.previous_lora_path:
            rank0_print(f"Loading lora weights from {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
        model = model.to(torch.bfloat16)
        # policy_model = policy_model.cpu()
        # del policy_model
        # PeftModel.from_pretrained(model)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        value_cls = obtain_value_cls(config)
        model = value_cls.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=1,
            trust_remote_code=True,
            pad_token_id=tokenizer.pad_token_id,
            # reduction=training_args.reduction,
            # cut_ratio=training_args.cut_ratio
        )
    if training_args.only_head:
        training_args.lora_enable = False
        for name, param in model.named_parameters():
            param.requires_grad = False
    if training_args.lora_enable:
        def module_map(s: str):
            modules = s.split(",")
            modules_to_wrap = [x + "_proj" for x in modules]
            return modules_to_wrap
        target_modules = module_map(training_args.target_modules) if training_args.target_modules is not None else None
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules if target_modules is not None else find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="SEQ_CLS"
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model = model.to(torch.bfloat16)
            if training_args.fp16:
                model = model.to(torch.float16)
        
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    rank0_print(model)
    if training_args.score_lr is None:
        training_args.score_lr = training_args.learning_rate
    for name, parameter in model.named_parameters():
        if "score" in name:
            parameter.requires_grad = True
        # if parameter.requires_grad:
        #     rank0_print(name, parameter.shape)
    dataset = obtain_dataset(data_args.data_path, tokenizer, even_value=data_args.even_value,
                             even_pair=data_args.even_pair, learn_rollout_value=data_args.learn_rollout_value)
    dataset = split_dataset(dataset, data_args.test_size)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    with add_proxy():
        # trainer = Trainer(
        trainer = ValueTrainer(
            model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        
        trainer.save_state()

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
    # non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    #     model.named_parameters()
    # )
    # if training_args.local_rank == 0 or training_args.local_rank == -1:
    #     model.config.save_pretrained(training_args.output_dir)
    #     rank0_print("Save score linear")
    #     # model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    #     torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))