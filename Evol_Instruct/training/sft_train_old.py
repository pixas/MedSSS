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

import os
import copy
from dataclasses import dataclass, field
import json

import logging
import pathlib

from pathlib import Path
from typing import Dict, Optional, Sequence, List
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
# from ming.model.modeling_phi import PhiForCausalLM
# from transformers.models.qwen2 import Qwen2ForCausalLM
import torch
import warnings
import transformers
from transformers import Trainer


from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import Dataset
from Evol_Instruct.utils.conversations import get_default_conv_template, SeparatorStyle
import pdb

import warnings
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from Evol_Instruct import client
from peft import LoraConfig, get_peft_model, PeftModel, LoftQConfig

import peft 

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
local_rank = None

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    freeze_backbone: Optional[bool] = field(default=False)
    
    # mix of lora arguments
    num_experts: Optional[int] = field(default=0)
    num_experts_per_token: Optional[int] = field(default=1)
    expert_selection: Optional[str] = field(default="top_k", metadata={"help": "top_k or sampling"})
    share_expert: Optional[bool] = field(default=False)
    output_logit_loss: Optional[int] = field(default=0, metadata={
        "help": "0 for not using logit loss; 1 for using absolute logit loss; \
            2 for using relative logit loss; 3 for using softmax logit loss;"
    })
    wrap_modules: Optional[List[str]] = field(default_factory=list)
    router_loss_coeff: Optional[float] = field(default="0.")
    add_identity_mapping: Optional[int] = field(default=0)
    
    # progressive params
    lora_name_or_path: Optional[str] = field(default=None)
    
    
    
    # other methods
    use_milora: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    train_data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    val_data_path: str = field(default=None,
                           metadata={"help": "Path to the validation data."})
    prompt_type: str = field(default="qwen",
                           metadata={"help": "prompt type"})
    is_base: bool = field(default=False, metadata={"help": "whether to use no-chat tuned model as the seed model"})

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
    molora_r: int = -1
    molora_alpha: int = -1
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_use_rs: bool = False
    wrap_ffn: bool = True
    wrap_attn: bool = True
    use_dora: Optional[bool] = field(default=False)
    use_mora: Optional[bool] = field(default=False)
    mora_type: Optional[int] = field(default=None)
    use_asylora: Optional[bool] = field(default=False)
    loraplus_lr_ratio: Optional[float] = field(
        default=None, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    use_loftq: Optional[bool] = field(default=False)
    
    # our method
    inference_path: Optional[int] = field(default=1)
    soft_select: Optional[bool] = field(default=False)
    output_oss_dir: str = field(default="s3://syj_test/checkpoint")
    
    wrap_ffn_layers: Optional[str] = field(default=None, metadata={
        "help": "please use [start_layer, end_layer] as the format where start_layer starts from 0 and end_layer is not included"
    })
    wrap_attn_layers: Optional[str] = field(default=None, metadata={
        "help": "please use [start_layer, end_layer] as the format where start_layer starts from 0 and end_layer is not included"
    })

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k and "experts" not in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    # mix of lora parameters
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, whether_wrap_ffn=True, whether_wrap_attn=True, layer_cls=Qwen2DecoderLayer):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'switch']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if not whether_wrap_ffn:
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), layer_cls) and isinstance(module, cls) and ("mlp" in name or "feed_forward" in name):
                continue
        if not whether_wrap_attn:
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), layer_cls) and isinstance(module, cls) and "self_attn" in name:
                continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_llama2(
    prompt_type,
    sources,
    tokenizer,
    is_base=False
) :
    # if is_base:
    #     prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    #     base_sources = [
    #         prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    #         for example in sources
    #     ]
    conv = get_default_conv_template(prompt_type).copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    if "role" in sources[0][0]:
        role_key = "role"
    else:
        role_key = "from"
    for i, source in enumerate(sources):
        if roles[source[0][role_key]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        if source[0]["value"] == "":
            source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence[role_key]]
                sentence[role_key] = "human" if j % 2 == 0 else "gpt"

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence[role_key]]
            assert role == conv.roles[j % 2], f"{i}, {sentence}, {role}, {j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(conversations)
    # Tokenize conversations
    # pdb.set_trace()
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # print(conversations[0])
    # Mask targets
    sep = "[/INST] "
    # print(tokenizer(conv.sep2))
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        conversation = "<s>" + conversation
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:cur_len] = IGNORE_TOKEN_ID
        cnt = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            # print("当前处理的对话：", rou)
            parts[0] += sep
            
            round_len = len(tokenizer(rou).input_ids) 
            

            instruction_len = len(tokenizer(parts[0]).input_ids) - 2



            # print("被mask的部分：",tokenizer.decode(target[cur_len:cur_len+instruction_len]))
            target[cur_len:cur_len+instruction_len] = (IGNORE_TOKEN_ID)
            
            cnt += 1
            cur_len += round_len

        target[cur_len:] = IGNORE_TOKEN_ID

    return dict(input_ids=input_ids, labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id))
def preprocess_llama3(
    prompt_type,
    sources,
    tokenizer,
    is_base=False
) :
    conv = get_default_conv_template(prompt_type).copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    if "role" in sources[0][0]:
        role_key = "role"
    else:
        role_key = "from"
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0][role_key]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        if source[0]["value"] == "":
            source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence[role_key]]
                sentence[role_key] = "human" if j % 2 == 0 else "gpt"

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # pdb.set_trace()
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()


    # Mask targets
    sep = "<|end_header_id|>\n\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split("<|eot_id|>")
        cur_len = 0
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if "<|start_header_id|>system<|end_header_id|>" in rou or "<|start_header_id|>user<|end_header_id|>" in rou:
                round_len = len(tokenizer(rou).input_ids) + 1
                target[cur_len:cur_len+round_len] = (IGNORE_TOKEN_ID)
                cur_len += round_len 
                
            else:
                parts = rou.split(sep)
                if len(parts) != 2:
                    break 
                parts[0] += sep 
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)
                target[cur_len:cur_len + instruction_len] = (IGNORE_TOKEN_ID)
                cur_len += round_len
            
           

        target[cur_len:] = IGNORE_TOKEN_ID
        
        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_TOKEN_ID
        #         print(f"WARNING: tokenization mismatch "
        #                     f"Round: {len(rounds)}, {cur_len} vs. {total_len}")

    return dict(input_ids=input_ids, labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id))
    
def preprocess(
    prompt_type,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    is_base = False
    # max_len: int,
    # system_message: str = "You are a helpful assistant."
) -> Dict:
    if prompt_type == 'llama2' or prompt_type == 'llama2_harm' or prompt_type == 'llama2_harm2' or prompt_type == 'mistral':
        return preprocess_llama2(prompt_type, sources, tokenizer, is_base=is_base)
    if prompt_type == 'llama3':
        return preprocess_llama3(prompt_type, sources, tokenizer, is_base=is_base)
    conv = get_default_conv_template(model_name=prompt_type).copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    system_message = conv.system

    # im_start = tokenizer.im_start_id
    # im_end = tokenizer.im_end_id
    im_start = 151644
    im_end = 151645
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    if "role" in sources[0][0]:
        role_key = "role"
    else:
        role_key = "from"

    for i, source in enumerate(sources):
        if roles[source[0][role_key]] != roles["human"]:
            source = source[1:]
        if source[0]["value"] == "":
            source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence[role_key]]
                sentence[role_key] = "human" if j % 2 == 0 else "gpt"
                
        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence[role_key]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(input_id))
        target += [IGNORE_TOKEN_ID] * (tokenizer.model_max_length - len(target))
        input_ids.append(input_id[:tokenizer.model_max_length])
        targets.append(target[:tokenizer.model_max_length])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 prompt_type: str):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(prompt_type, sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    attention_mask=self.attention_mask[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 prompt_type: str,
                 is_base: bool = False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.prompt_type = prompt_type

        rank0_print("Loading data...")
        list_data_dict = client.read(data_path)
        rank0_print("Loading total {} instances...".format(len(list_data_dict)))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict
        self.is_base = is_base

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        # print(sources[0])
        data_dict = preprocess(self.prompt_type, [e["conversations"] for e in sources],
            self.tokenizer, is_base=self.is_base)
        # print(data_dict["attention_mask"][0])
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             attention_mask=data_dict["attention_mask"][0])
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        # input_ids, labels = tuple([torch.tensor(instance[key]).long() for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = [], []
        for instance in instances:
            instance_len = instance["input_ids"].ne(self.tokenizer.pad_token_id).sum(-1)
            input_ids.append(instance["input_ids"][:instance_len].long())
            labels.append(instance["labels"][:instance_len].long())

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class MultiplePathDataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    inference_path: int  = 1

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = [], []
        for instance in instances:
            instance_len = instance["input_ids"].ne(self.tokenizer.pad_token_id).sum(-1)
            input_ids.append(instance["input_ids"][:instance_len].long())
            labels.append(instance["labels"][:instance_len].long())

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        
        input_ids, labels, attention_mask = self._repeat_input(input_ids, labels, attention_mask)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    
    def _repeat_input(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        if self.inference_path == 1:
            return input_ids, labels, attention_mask
        input_ids = input_ids.repeat(self.inference_path, 1)
        labels = labels.repeat(self.inference_path, 1)
        attention_mask = attention_mask.repeat(self.inference_path, 1) 
        
        return input_ids, labels, attention_mask


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                inference_path:int = 1) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.train_data_path,
                                prompt_type=data_args.prompt_type,
                                is_base=data_args.is_base)
    eval_dataset = dataset_cls(tokenizer=tokenizer,
                                         data_path=data_args.val_data_path,
                                prompt_type=data_args.prompt_type, is_base=data_args.is_base) if data_args.val_data_path is not None else None
    
    if inference_path > 1:
        data_collator = MultiplePathDataCollatorForSupervisedDataset(
            tokenizer=tokenizer, inference_path=inference_path
        )
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
                
def freeze_layer(model, wrap_ffn_layers, wrap_attn_layers):
    attn_layers = [int(x) for x in wrap_attn_layers[1:-1].split(",")] if wrap_attn_layers is not None else []
    ffn_layers = [int(x) for x in wrap_ffn_layers[1:-1].split(",")] if wrap_ffn_layers is not None else []
    for name, param in model.named_parameters():
        if "mlp" in name or "self_attn" in name:
            
            layer_idx = int(name.split(".")[4])
            if "mlp" in name and "lora" in name:
                # this is the trainable param
                if len(ffn_layers) == 0:
                    # nothing happens, a trivial case
                    continue 
                if layer_idx < ffn_layers[0] or layer_idx >= ffn_layers[1]:
                    param.requires_grad = False
            elif "self_attn" in name and "lora" in name:
                if len(attn_layers) == 0:
                    continue
                if layer_idx < attn_layers[0] or layer_idx >= attn_layers[1]:
                    param.requires_grad = False
                
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)


    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **bnb_model_from_pretrained_args)
    decoder_layer_cls = None


    # model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **bnb_model_from_pretrained_args)
    model.config.use_cache = False
    

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable or model_args.num_experts > 1:
        def module_map(s: str):
            modules = s.split(",")
            modules_to_wrap = [x + "_proj" for x in modules]
            return modules_to_wrap
        target_modules = module_map(training_args.target_modules) if training_args.target_modules is not None else None
        if peft.__version__ == "0.9.0":
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules if target_modules is not None else find_all_linear_names(model, whether_wrap_ffn=training_args.wrap_ffn if model_args.num_experts <= 1 else False, whether_wrap_attn=training_args.wrap_attn, layer_cls=decoder_layer_cls),
                # target_modules=['q_proj', 'v_proj'],
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
                use_rslora=training_args.lora_use_rs,
                use_dora=training_args.use_dora,
                # loftq_config=LoftQConfig(loftq_bits=4) if training_args.use_loftq else None
            )
        else:
            if training_args.lora_use_rs:
                warnings.warn("You set use_rslora as True when using an unsupported peft version; try `pip install peft --upgrade` to fix it.")
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules if target_modules is not None else find_all_linear_names(model, whether_wrap_ffn=training_args.wrap_ffn if model_args.num_experts <= 1 else False, whether_wrap_attn=training_args.wrap_attn, layer_cls=decoder_layer_cls),
                # target_modules=['q_proj', 'v_proj'],
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
                use_dora=training_args.use_dora,
                # loftq_config=LoftQConfig(loftq_bits=4) if training_args.use_loftq else None
            )
    if training_args.lora_enable:
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")

            # model_cls, decoder_layer_cls = get_model_class_from_path(model_args.model_name_or_path)
            # model = get_mora_model(model, lora_config, decoder_type=decoder_layer_cls, wrap_modules=lora_config.target_modules, mora_type=training_args.mora_type)
            # training_args.mora = True 
        

        model = get_peft_model(model, lora_config)

    
  

    rank0_print(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False)
    # pdb.set_trace()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              inference_path=training_args.inference_path)
    # print(model)

    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         rank0_print(n)

    trainer = Trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    # model.tokenizer.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        if not os.path.exists(os.path.join(training_args.output_dir, "adapter_config.json")):
            lora_config.save_pretrained(training_args.output_dir)
        if not os.path.exists(os.path.join(training_args.output_dir, "tokenizer.json")):
            tokenizer.save_pretrained(training_args.output_dir)
    elif model_args.freeze_backbone:
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    train()
