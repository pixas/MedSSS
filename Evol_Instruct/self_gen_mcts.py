from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from Evol_Instruct import client, logger
from peft import PeftModel
import torch
from Evol_Instruct.MCTS import *

from Evol_Instruct.MCTS.tree_register import tree_registry
import argparse
# print(cd_pipeline_registry)
import json

import re
import random 
import os
# from Evol_Instruct.models.modeling_value_llama import  obtain_value_cls
from Evol_Instruct.models.openai_access import batch_call_chatgpt
from Evol_Instruct.utils.utils import extract_answer, get_chunk
from Evol_Instruct.prompts.prompt_template import orm_teacher_prompt

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from Evol_Instruct.models.vllm_support import VLLMServer, get_vllm_model, vllm_clean_generate

from Evol_Instruct.MCTS.tree import MCTSConfig, MedMCTSNode


class LogitBiasProcess(LogitsProcessor):
    def __init__(self, activate_token_list: list[int] = None, activate_scale=100):
        self.activate_token_list = activate_token_list
        self.activate_scale=activate_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # logger.info(scores.shape, input_ids.shape)
        for id_ in self.activate_token_list:
            if scores.dim() == 2:
                scores[:, id_] += self.activate_scale
            else:
                scores[id_] += self.activate_scale
        return scores



def set_tokenizer(tokenizer):
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer    




def mcts_generate(args, server, items, **kwargs):

    config = kwargs.pop("config")
    # lora_request = kwargs.pop("lora_request", None)
    value_function = kwargs.pop("value_function", None)
    item = items[0]
    # if args.repeat_try >= 1:
    if isinstance(getattr(config.expand, "autostep", "step"), int):
        mcts_cls = LSMCTS
    elif isinstance(getattr(config.expand, "autostep", "step"), str):
        if getattr(config, "manual", False):
            mcts_cls = MMCTS
        else:
            mcts_cls = tree_registry.get(getattr(config, 'mcts_cls', 'MCTS'), MCTS)
            # if getattr(config, 'mcts_cls', 'MCTS')
            # mcts_cls = MCTS
    # logger.info(f"Setting MCTS class to {mcts_cls}")
    tree = mcts_cls(item, server, config, value_function=value_function, training=True, first_round=args.iter == 1)
    # tree = LSMCTS(item, server, config, value_function=value_function)
    root = tree.run()
    # print(tree.obtain_leaves(root), flush=True)
    # should return some leaves, first, return correct trajectories, serving as SFT pool
    correct_leaves: list[MedMCTSNode] = tree.obtain_correct_leaves() # list[str]
    repeat_try = getattr(config, "repeat_try", 0)
    while len(correct_leaves) == 0 and repeat_try > 0:
        tree = mcts_cls(item, server, config, value_function=value_function)
        root = tree.run()
        correct_leaves = tree.obtain_correct_leaves()
        repeat_try -= 1
    # then, return incorrect trajectories, serving as bad value function pool
    incorrect_leaves: list[MedMCTSNode] = tree.obtain_incorrect_leaves() # list[str]
    # finally select intermediate nodes, serving as anothher value function pool 
    # question: how to select
    if args.intermediate_select == 'all':
        inter_nodes: list[MedMCTSNode] = root.select_all_intermediate_nodes()
    elif args.intermediate_select == 'representative':
        inter_nodes: list[MedMCTSNode] = root.select_representative_nodes()
    # intermediate_trajectories: list[str] = [node.obtain_reasoning_steps() for node in inter_nodes]
    # construct the sft pool
    # correct_output = [leaf.obtain_reasoning_steps()[0] for leaf in correct_leaves]
    # correct_output = list(set(correct_output))
    # value function pool 
    # save_item = 
    output = {
        "id": item['id'],
        "question": item['conversations'][0]['value'],
        "pos": [(leaf.reasoning_chain(), leaf.value, leaf.simulation_score) for leaf in correct_leaves],
        "neg": [(leaf.reasoning_chain(), leaf.value, leaf.simulation_score) for leaf in incorrect_leaves],
        # "inter": [(leaf.reasoning_chain(), leaf.value, leaf.simulation_score) for leaf in inter_nodes],
        "answer": item['answer_idx'],
        "original_dataset": getattr(item, 'dataset', None)
    }
    # exit(0)
    # value_pool = [{"id": item['id'],
    #                "trace": leaf.value_chain(),
    #     "question": root.problem,
    #                "output": leaf.obtain_reasoning_steps()[0],
    #                "label": leaf.value} for leaf in correct_leaves + incorrect_leaves + inter_nodes]
    return output

    
def post_process_output(batch_data):
    def rule_filter(x):
        if isinstance(x, list):
            return [rule_filter(y) for y in x]
        output = x
        if any(x.startswith(forbid) for forbid in ["Here is", "Based on", "Here's"]):
            output = "\n\n".join(x.split("\n\n")[1:])
        elif any(forbid in x for forbid in ["I'd like", "I'd be", "I'm", "I would"]):
            output = "\n\n".join(x.split("\n\n")[1:])
        if any(forbid in x for forbid in ["I hope"]):
            output = "\n\n".join(x.split("\n\n")[:-1])
            
        return output
    new_data = [rule_filter(x) for x in batch_data]
    return new_data

def saving_process(path):
    json_output_path = path.replace(".jsonl", ".json")
    jsonl_data = client.read(path)
    # logger.info(path)
    # logger.info(jsonl_data)
    # logger.info(json_output_path)
    client.write(jsonl_data, json_output_path, indent=2)
    # os.remove(path)
    client.remove(path)

def save_helper(data, path):
    if path.startswith("s3://"):
        path = path.replace("s3://syj_test", "/mnt/petrelfs/jiangshuyang.p/oss")
    # logger.info(path)
    with open(path, 'a') as f:
        f.write(data)
        
def process_dataset(args, data):
    # tokenizer = model.get_tokenizer()
    new_data = []
    config = client.read(args.config)
    config = MCTSConfig(config)

    
    # probe whether there exists a os.path.dirname(args.output_path)/sft_combined_{iter}.json
    # if so, read the data, and filter those already processed
    if os.path.exists(os.path.join(os.path.dirname(args.output_path), f"sft_{args.iter}.json")):
        processed_data = client.read(os.path.join(os.path.dirname(args.output_path), f"sft_{args.iter}.json"))
        processed_ids = set([x['id'] for x in processed_data])
        data = [item for item in data if item['id'] not in processed_ids]

    
    # if args.shots > 0:
    #     data = data[args.shots:]
    
    data = get_chunk(data, args.num_chunks, args.chunk_idx)
    processed_ids = set()
    if args.resume:
        # if os.path.exists(args.output_path):
        # logger.debug(args.output_path)
        if client.exists(args.output_path):
            # with open(args.output_path, 'r') as f:
            #     for line in f:

            #         new_data.append(json.loads(line))
            new_data = client.read(args.output_path)
            # obtain all processed id 
            processed_ids = set([x['id'] for x in new_data])
            # start_index = int(new_data[0]['id'].split("_")[-1])
            # end_index = int(new_data[-1]['id'].split("_")[-1])
            # process_num = end_index - start_index + 1
            logger.debug(len(processed_ids))
            # data = data[process_num:]
            logger.info("Resume from already processed file")
        elif client.exists(args.output_path.replace("jsonl", "json")):
            new_data = client.read(args.output_path.replace("jsonl", "json"))
            processed_ids = set([x['id'] for x in new_data])
            # start_index = int(new_data[0]['id'].split("_")[-1])
            # end_index = int(new_data[-1]['id'].split("_")[-1])
            # process_num = end_index - start_index + 1
            # data = data[process_num:]
            logger.debug(len(processed_ids))
            logger.info("Resume from already processed file")
        else:
            logger.info("No processed data yet.")
            # client.write([], args.value_output_path)
    else:
        client.write([], args.output_path)
        # with open(args.output_path, 'w') as f:
        #     f.write('')
        # client.write([], args.value_output_path)
        # with open(args.dpo_output_path, 'w') as f:
        #     f.write('')
            
    data = [item for item in data if item['id'] not in processed_ids]        
    
            
    logger.info(f"Data to be processed number: {len(data)}")
    logger.info(f"Data will be saved to {args.output_path}")
    if len(data) == 0:
        logger.info("No data to be processed, exit normally")
        logger.info(args.output_path)
        if client.exists(args.output_path):
            logger.info(f"Remove {args.output_path}")
            client.remove(args.output_path)
        elif client.exists(args.output_path.replace("/mnt/petrelfs/jiangshuyang.p/oss", "s3://syj_test")):
            file = args.output_path.replace("/mnt/petrelfs/jiangshuyang.p/oss", "s3://syj_test")
            logger.info(f"Remove {file}")
            client.remove(file)
        exit(0)


    if args.lora_path is None or args.lora_path == "None":
        lora_path = None
    else:
        logger.info(f"Loading LoRA weights from {args.lora_path}")
        lora_path = args.lora_path
        

    if args.value_function != ["None"] and args.value_function is not None:
        # config.value_function = args.value_function
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        config = AutoConfig.from_pretrained(args.model_path)
        # value_cls = AutoModelFor(config)
        value_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=1,
            trust_remote_code=True,
            pad_token_id=tokenizer.pad_token_id
        )
        for vf in args.value_function:
            value_model = PeftModel.from_pretrained(value_model, vf)
            value_model = value_model.merge_and_unload()
            logger.info(f"Loading value model from {vf}")
        # value_model = PeftModel.from_pretrained(model_base, args.value_function)
        value_model = value_model.to(torch.float16)
        gpu_memory = 0.45
    else:
        gpu_memory = 0.9
        value_model = None
    # model, lora_request = get_vllm_model(args.model_path, lora_path=lora_path,
    #                                      gpu_memory_usage=gpu_memory)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    
    tokenizer = set_tokenizer(tokenizer)
    server = VLLMServer(url=args.url, tokenizer=tokenizer, model=args.model_path, offline=True if args.url is None else False, lora_path=lora_path, gpu_memory_usage=gpu_memory)
    # tokenizer = model.get_tokenizer()
    
    for idx in trange(0, len(data), args.batch_size):
        items = data[idx:idx+args.batch_size]
        try:
            output = mcts_generate(args, server, items, config=config, value_function=value_model,)
        
            save_helper(json.dumps(output) + "\n", args.output_path)
        except Exception as e:
            print(e)
            pass
        
        #
    saving_process(args.output_path)
    # saving_process(args.value_output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--value_output_path', type=str, default=None)

    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    
    parser.add_argument("--url", type=str, default=None)
    
    

    # sc parameter
    parser.add_argument('--iter', type=int, default=1)
    # parser.add_argument('--repeat_try', type=int, default=1)
    parser.add_argument("--config", type=str, default="Evol_Instruct/config/default_config.json")
    parser.add_argument('--lora_path', type=str, default=None, help="Path to the lora-tuned model.")
    parser.add_argument('--intermediate_select', type=str, default='all', help="Select intermediate nodes.")
    parser.add_argument('--value_function', type=str, default=None, nargs="+")
    
    args = parser.parse_args()
    if args.value_output_path is None:
        output_name = args.output_path.split("/")[-1]
        args.value_output_path = args.output_path.replace(output_name, f"value_{output_name}")
    
    data = client.read(args.data_path)
    torch.set_default_device("cuda")
    args.batch_size = 1 # MCTS only supports inferring one sample at a time
    
    process_dataset(args, data)
    