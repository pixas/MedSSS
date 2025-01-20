from copy import deepcopy
from Evol_Instruct import client, logger

import argparse
from multiprocessing import Pool, cpu_count, Lock
import json 
import os
import fcntl  # 用于文件锁
from tqdm import tqdm

from Evol_Instruct.models.openai_access import call_chatgpt 


lock = Lock()
def write_jsonl(file_path, obj):
    with open(file_path, 'a') as f:
        # 使用文件锁来确保写入的同步性
        with lock:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(obj) + '\n')
            fcntl.flock(f, fcntl.LOCK_UN)


def process_obj(args_tuple):
    cur_obj, model = args_tuple
    result = call_chatgpt(model, cur_obj['conversations'][0]['value'])
    new_obj = deepcopy(cur_obj)
    new_obj['conversations'][1]['value'] = result[0]
    return new_obj

def saving_process(path):
    json_output_path = path.replace(".jsonl", ".json")
    jsonl_data = client.read(path)
    client.write(jsonl_data, json_output_path, indent=2)
    os.remove(path)

def main(args,):
    data = client.read(args.data)
    
    new_data = []
    if args.sample_num != -1:
        data = data[:args.sample_num]
    if args.shots > 0:
        data = data[args.shots:]
    if args.resume:
        if os.path.exists(args.output_path):
            with open(args.output_path, 'r') as f:
                for line in f:

                    new_data.append(json.loads(line))
            start_index = int(new_data[-1]['id'].split("_")[-1])
            data = data[start_index+1:]
            logger.info("Resume from already processed file")
        elif os.path.exists(args.output_path.replace("jsonl", "json")):
            new_data = client.read(args.output_path.replace("jsonl", "json"))
            start_index = int(new_data[-1]['id'].split("_")[-1])
            data = data[start_index+1:]
            logger.info("Resume from already processed file")
        else:
            logger.info("No processed data yet.")

    else:
        with open(args.output_path, 'w') as f:
            f.write('')

            
    logger.info(f"Data to be processed number: {len(data)}")
    if len(data) == 0:
        logger.info("No data to be processed, exit normally")
        exit(0)
        
    if args.num_process != 1:
        if args.num_process == -1:
            num_processes = min(cpu_count(), len(data))
        else:
            num_processes = args.num_process

        with Pool(num_processes) as pool:
            for result in tqdm(pool.imap(process_obj, [(cur_obj, args.model,) for cur_obj in data]), desc='GPT-4 Distilling', total=len(data)):
                # if result is not None:
                #     for obj in result:
                new_obj = result
                write_jsonl(args.output_path, new_obj)
                
    saving_process(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/evol_instruct.jsonl')
    parser.add_argument('--output_path', type=str, default='data/evol_instruct_gpt.jsonl')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sample_num', type=int, default=-1)
    parser.add_argument('--shots', type=int, default=0)
    args = parser.parse_args()
    main(args)