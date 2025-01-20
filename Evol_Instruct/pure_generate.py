import json
import random

from Evol_Instruct.models.vllm_support import get_vllm_model, vllm_generate
from Evol_Instruct.models.openai_access import call_chatgpt
from Evol_Instruct.prompts.prompt_template import gen_w_prior_prompt
from transformers import AutoTokenizer
import argparse 
from copy import deepcopy
from tqdm import tqdm
import os
from Evol_Instruct.utils.utils import client
from multiprocessing import Pool, cpu_count, Lock
import fcntl  # 用于文件锁
import math
from typing import List
from Evol_Instruct.utils.parser_creater import get_general_parser, get_sampling_parser
from Evol_Instruct.utils.my_logger import logger
from Evol_Instruct.utils.utils import get_chunk
from Evol_Instruct.utils.dataset import EvolDataloader, EvolDataset
import random



lock = Lock()


def post_process(list_of_instruction: List[str]):
    process_words = ['#Rewritten Prompt#', '#Created Prompt#']
    new_instruction = []
    for instruction in list_of_instruction:
        for word in process_words:
            if instruction.startswith(word):
                instruction = instruction.split(word)[1].strip(":").strip().strip("\n").strip()
                
        new_instruction.append(instruction)
    return new_instruction




# def read_jsonl(file_path):
# 	objs = []
# 	with open(file_path, 'r') as f:
# 		for line in f:
# 			objs.append(json.loads(line))
# 	return objs

def write_jsonl(file_path, obj):
    with open(file_path, 'a') as f:
        # 使用文件锁来确保写入的同步性
        with lock:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(obj) + '\n')
            fcntl.flock(f, fcntl.LOCK_UN)


def simple_filter(instructions):
    filtered_instructions = []
    for instruction in instructions:
        # some rule:
        # 1. to create
        # here is, here's 
        if "to create" in instruction.lower() or "version" in instruction.lower() or "here's" in instruction.lower():
            continue 
        filtered_instructions.append(instruction)
    return filtered_instructions



def get_generate_prompt(instructions, priors):
    prompts = [gen_w_prior_prompt.format(question=instruction, reference=prior) for instruction, prior in zip(instructions, priors)]
    return prompts


def process_os_obj(args_tuple):
    cur_objs, model, tokenizer, prompt_type, max_rounds, sampling_args = args_tuple 
    sampling_kwargs = {key: value for key, value in sampling_args._get_kwargs() if value is not None and (value)}
    generated_objs = []
    if any(len(cur_obj['conversations']) > 2 for cur_obj in cur_objs):

        raise NotImplementedError

    else:
        instructions = [cur_obj['conversations'][0]['value'].strip() for cur_obj in cur_objs]
        priors = [cur_obj['reference'] for cur_obj in cur_objs]
        generate_prompts = get_generate_prompt(instructions, priors)

        answers = vllm_generate(model, tokenizer, generate_prompts, True, **sampling_kwargs)
        for i in range(min(len(generate_prompts), len(answers))):
        # for i, question in enumerate(evol_instructions):
            new_obj = deepcopy(cur_objs[i])
            new_obj['conversations'][0]['value'] = instructions[i]
            new_obj['conversations'][1]['value'] = answers[i]

            generated_objs.append(new_obj)
    return generated_objs
    

def generate_answer_with_prior(args, sampling_args):
    all_objs = client.read(args.input_data)
    
    logger.info("Read input data over...")
    sample_selected = all_objs[args.sample_idx]

    if args.sample_num != -1:
        all_objs = all_objs[:args.sample_num]
    start_idx = 0

    merge_file = "/".join(args.output_data.split("/")[:-1]) + "/merge.jsonl"
    if args.resume_merge_file and os.path.exists(merge_file):	
        with open(merge_file, 'r') as f:
            last_line = f.readlines()[-1]
            last_obj = json.loads(last_line)
            last_id = int(last_obj['id'].split("_")[-1])
            start_idx = last_id + 1 
            logger.info(f"Resume from merged file, start_idx: {start_idx}")
            # print("Successfully resume from merged file, set args.resume to False")
            # args.resume = False
    all_objs = all_objs[start_idx:]



    chunk_size = math.ceil(len(all_objs) / args.num_chunks)
    all_objs = get_chunk(all_objs, args.num_chunks, args.chunk_idx)
    begin_idx = start_idx + chunk_size * args.chunk_idx  # integer division // args.num_chunks
    start_idx = 0
    # evol_objs = []

    if args.resume and os.path.exists(args.output_data) and (not args.clean_outputs):
        with open(args.output_data, 'r') as f:
            all_lines = f.readlines()
            if len(all_lines) >= 1:
                last_line = all_lines[-1]
                last_obj = json.loads(last_line)
                last_id = int(last_obj['id'].split("_")[-1])
                start_idx = last_id + 1 - begin_idx

        logger.info(f"Resume from the current file, start_idx: {start_idx}")
    else:
        with open(args.output_data, 'w') as fw:
            fw.write("")
    all_objs = all_objs[start_idx:]
    if args.sample_idx != -1:
        logger.info(f"Only process specified data sample {args.sample_idx} for debug")
        all_objs = [sample_selected]
    
    if args.num_process != 1:
        raise NotImplementedError
    
    else:
        if len(all_objs) > 0:
            evol_dataset = EvolDataset(all_objs)
            evol_dataloader = EvolDataloader(evol_dataset, args.batch_size)
            model = get_vllm_model(args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            with tqdm(total=len(evol_dataloader), desc=f"Generate", ncols=100, 
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                for cur_objs in tqdm(evol_dataloader, total=len(evol_dataloader)):
                    
                    generated_objs = process_os_obj((cur_objs, model, tokenizer, args.prompt_type, args.max_rounds, sampling_args))
                    for obj in generated_objs:
                        
                        write_jsonl(args.output_data, obj)
                    pbar.update(1)
            pass
    logger.info("Process Over...")

if __name__ == "__main__":
    general_parser = get_general_parser()
    sampling_parser = get_sampling_parser()
    
    general_parser.add_argument('--prior_file', type=str, default=None)
    general_args, remaining_args = general_parser.parse_known_args()
    sampling_args, _ = sampling_parser.parse_known_args()
    
    generate_answer_with_prior(general_args, sampling_args)