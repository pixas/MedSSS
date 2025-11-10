

import time
from datasets import Dataset, load_from_disk, DatasetDict
from queue import PriorityQueue, Queue
import shutil
import os 
from tqdm import tqdm
from Evol_Instruct import client, logger
import tempfile
import uuid
import pathlib
from dataclasses import dataclass, field
import torch
from accelerate import PartialState
from Evol_Instruct.MCTS.tree_node import MedMCTSNode
from Evol_Instruct.training.value_train import uniform_sample, shuffle_list, print_on_main

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from functools import partial

from Evol_Instruct.utils.utils import add_proxy

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
def return_prompt_and_responses(samples, tokenizer):
    # print(list(samples.keys()))
    return {
        "prompt": [tokenizer.apply_chat_template(sample,
                                                 tokenize=False,
                                                 add_generation_prompt=True) for sample in samples['prompt']],
        "chosen": [sample[0]['content'] for sample in samples['chosen']],
        "rejected": [sample[0]['content'] for sample in samples['rejected']]
    }

def trace_value(node):
    value_list = []
    cur = node
    while cur:
        value_list.append(cur.value)
        cur = cur.parent
    value_list = value_list[:-1]
    return min(value_list)

def ls_obtain_dataset(data, tokenizer, training_args):
    # use level search to group the step-level DPO data
    # chosen and rejected only happens with the same depth
    # also, should try to make that the parent of chosen node should be different from the parent of rejected node
    finish_cnt = 2
    new_data = []
    for item in tqdm(data, desc="Processing data", disable=local_rank != 0, total=len(data)):
        if len(item['pos']) == 0 or len(item['neg']) == 0:
            continue 
        tree = MedMCTSNode.from_list(item)
        queue = Queue()
        queue.put(tree)
        visited = set()
        finished_set = set()
        finish_nodes = []
        while not queue.empty():
            node_list = []
            while not queue.empty():
                node = queue.get()
                if node in visited:
                    continue
                visited.add(node)
                node_list.append(node)
            # select chosen and rejected from the node_list 
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    if abs(node_list[i].value - node_list[j].value) > 0.8 and node_list[i].parent != node_list[j].parent and len(node_list[i].parent.children) > 1 and len(node_list[j].parent.children) > 1:
                        large_node = max(node_list[i], node_list[j], key=lambda x: x.value)
                        small_node = min(node_list[i], node_list[j], key=lambda x: x.value)
                        new_data.append({
                            "chosen": [{"role": "assistant", "content": large_node.obtain_reasoning_steps()[0] }],
                            "rejected": [{"role": "assistant", "content": small_node.obtain_reasoning_steps()[0] }],
                            "prompt": [{"role": "user", "content": large_node.problem}]
                        })
                        if large_node.children == [] and small_node.children == []:
                            # a finish node has been added to training list, mark it as visited
                            finished_set.add((large_node, small_node))
                            # finished_set.add(small_node)
            for node in node_list:
                # if node.parent is not None and node.parent.value - node.value > 0.3:
                #     new_data.append({
                #         "chosen": [{"role": "assistant", "content": node.parent.obtain_reasoning_steps()[0] }],
                #         "rejected": [{"role": "assistant", "content": node.obtain_reasoning_steps()[0] }],
                #         "prompt": [{"role": "user", "content": node.parent.problem}]
                #     })
                for child in node.children:
                    if child.type == 'Finish':
                        finish_nodes.append(child)
                    queue.put(child)
        # finish pair 
        finish_node_trace = [trace_value(node) for node in finish_nodes]
        correct_finish_nodes = [(finish_node_trace[i], finish_node) for i, finish_node in enumerate(finish_nodes) if finish_node.value == 1.0]
        
        incorrect_finish_nodes = [(finish_node_trace[i], finish_node) for i, finish_node in enumerate(finish_nodes) if finish_node.value == 0.0]
        
        correct_finish_nodes = sorted(correct_finish_nodes, key=lambda x: x[0], reverse=True)
        incorrect_finish_nodes = sorted(incorrect_finish_nodes, key=lambda x: x[0], reverse=False)

        # finish_nodes = sorted(finish_nodes, key=lambda x: x.value, reverse=True)
            
        for i in range(len(correct_finish_nodes)):
            for j in range(len(incorrect_finish_nodes)):


                larger_trace, large_node = correct_finish_nodes[i]
                smaller_trace, small_node = incorrect_finish_nodes[j]
                

                
                if (large_node, small_node) in finished_set:
                    continue
                if (larger_trace - smaller_trace) <= 0.8:
                    continue
                new_data.append({
                    "chosen": [{"role": "assistant", "content": large_node.obtain_reasoning_steps()[0] }],
                    "rejected": [{"role": "assistant", "content": small_node.obtain_reasoning_steps()[0] }],
                    "prompt": [{"role": "user", "content": large_node.problem}]
                })
    return new_data


def dfs_obtain_dataset(data, tokenizer, training_args):
    train_pair_per_instance = -1
    new_data = []
    for item in data:
        if len(item['pos']) == 0 or len(item['neg']) == 0:
            continue 
        tree = MedMCTSNode.from_list(item)
        # for each node, if it has more than 1 child, select the highest value and lowest value 
        # node = tree
        stack = [tree]
        priority_queue = PriorityQueue()
        visited = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            # print_on_main(len(stack), flush=True)
            if len(node.children) > 1:
                higher_node = max(node.children, key=lambda x: x.value)
                lower_node = min(node.children, key=lambda x: x.value)
                if higher_node.value > lower_node.value:
                    if len(higher_node.children) == 0:
                        # a finish node has been added to training list, mark it as visited
                        visited.add(higher_node)
                    if len(lower_node.children) == 0:
                        # a finish node has been added to training list, mark it as visited
                        visited.add(lower_node)
                    new_data.append({
                        "chosen": [{"role": "assistant", "content": higher_node.obtain_reasoning_steps()[0] }],
                        "rejected": [{"role": "assistant", "content": lower_node.obtain_reasoning_steps()[0] }],
                        "prompt": [{"role": "user", "content": node.problem}]
                    })
                stack.extend(node.children)
            elif len(node.children) == 1:
                # only one child, select the child 
                stack.append(node.children[0])
                
            else:
                # enqueue priority_queue by the minimum value from root to current
                value_list = []
                cur = node
                while cur:
                    value_list.append(cur.value)
                    cur = cur.parent
                value_list = value_list[:-1]
                priority_queue.put((-min(value_list), node))

        # get first train_pair_per_instance items and last train_pair_per_instance from the queue
        queue = priority_queue.queue
        correct_node = [node for val, node in queue if node.value == 1.0]
        incorrect_node = [node for val, node in queue if node.value == 0.0]
        # shuffle_list(correct_node)
        # shuffle_list(incorrect_node)
        # get first train_pair_per_instance items and last train_pair_per_instance from the queue
        if train_pair_per_instance == -1:
            train_pair_per_instance = min(len(correct_node), len(incorrect_node))
        else:
            train_pair_per_instance = min(min(len(correct_node), len(incorrect_node)), train_pair_per_instance)

        for i in range(train_pair_per_instance):
            new_data.append(
                {
                    "chosen": [{"role": "assistant", "content": correct_node[i].obtain_reasoning_steps()[0] }],
                    "rejected": [{"role": "assistant", "content": incorrect_node[-i-1].obtain_reasoning_steps()[0] }],
                    "prompt": [{"role": "user", "content": correct_node[i].problem}],
                }
                
            )
        
    return new_data 



def obtain_dataset_each(data, tokenizer, training_args, script_args):
    if script_args.data_process_method == "dfs":
        new_data = dfs_obtain_dataset(data, tokenizer, training_args)
    elif script_args.data_process_method == "ls":
        new_data = ls_obtain_dataset(data, tokenizer, training_args)
    
            
    dataset = Dataset.from_list(new_data)
    with PartialState().local_main_process_first():
        dataset = dataset.filter(lambda x: len(tokenizer.apply_chat_template(x["prompt"]+x['chosen'], tokenize=True)) <= training_args.max_length and len(tokenizer.apply_chat_template(x["prompt"]+x['rejected'], tokenize=True)) <= training_args.max_length, num_proc=training_args.dataset_num_proc)
    # original_columns = dataset.column_names
    # process_func = partial(return_prompt_and_responses, tokenizer=tokenizer)
    # dataset = dataset.map(process_func, batched=True, remove_columns=original_columns)
    return dataset


def obtain_dataset(data_path, tokenizer, training_args, script_args):
    total_data = client.read(data_path)
    split_ratio = script_args.test_split_ratio
    # shuffle_list(total_data)
    test_num = int(len(total_data) * split_ratio)
    test_data = uniform_sample(total_data, test_num)
    train_data = [item for item in total_data if item not in test_data]
    # train_data = total_data[:-test_num]
    # test_data = total_data[-test_num:]
    
    train_dataset = obtain_dataset_each(train_data, tokenizer, training_args, script_args)
    test_dataset = obtain_dataset_each(test_data, tokenizer, training_args, script_args)
    
    return {"train": train_dataset, "test": test_dataset}

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer    


def train_dpo(config, model, tokenizer, dataset):
    trainer = DPOTrainer(config, model, tokenizer)
    trainer.train(dataset)
    return trainer

def split_dataset(dataset: Dataset, test_size=0.1):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split



# ModelConfig
# 
# class ModelConfiguration(ModelConfig):
@dataclass
class DPOScriptArguments(ScriptArguments):
    dataset_name: str = field(metadata={"help": "The name of the dataset to use."}, default=None)
    data_path: str = field(metadata={"help": "Path to the data file."}, default=None)
    test_split_ratio: float = field(metadata={"help": "Ratio of the test split."}, default=0.1)
    
    # if load a lora-tuned sft model
    tuned_lora_path: list[str] = field(metadata={"nargs": "+", "help": "Path to the lora-tuned model."}, default_factory=list)
    data_process_method: str = field(metadata={"help": "Method to process the data."}, default="dfs")

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    model_dtype = model.dtype
    print_on_main(script_args.tuned_lora_path)
    if script_args.tuned_lora_path is not None and script_args.tuned_lora_path != ["None"]:
        for lora_path in script_args.tuned_lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print_on_main(f"Load lora weights from {lora_path}")
        model = model.to(model_dtype)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name)
    state = PartialState()
    temp_dir = None

    dataset = obtain_dataset(script_args.data_path, tokenizer, training_args, script_args)
    dataset = DatasetDict(dataset)
    assert script_args.dataset_train_split in dataset, f"{script_args.dataset_train_split} not in dataset"
    ##########
    # Training
    ################
    with add_proxy():
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            # processing_class=tokenizer,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        # trainer.train()
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)
