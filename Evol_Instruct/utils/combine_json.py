import json 
import os
import sys 
import re 
# print(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from Evol_Instruct.utils.utils import client 
import argparse
def combine_json_data(data_path, save_path, prefix='sft', num=-1, only_iter=False):
    # list all files that start wth "dpo_"
    files = client.listdir(data_path)
    # print(files)
    pattern = rf'^{prefix}_\d+\.json$'
    files = [f for f in files if f.startswith(f"{prefix}_")]
    # print(files)
    if only_iter:
        files = [f for f in files if re.match(pattern, f)]
    else:
        files = [f for f in files if f.endswith(".json")]
    files.sort()
    # print(files)
    data = []
    if num == -1:
        num = len(files)
    for f in files[:num]:
        file = os.path.join(data_path, f)
        # print(file)
        data.extend(client.read_json(file))
    print(f"Combine {len(data)} data from {num} files")
    if os.path.exists(save_path):
        already_saved_data = client.read_json(save_path)
        data.extend(already_saved_data)
    client.write(data, save_path, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform jsonl to json")
    parser.add_argument("--data_path", type=str, help="The path of the json file")
    parser.add_argument('--save_path', type=str, help="The path of the new json file")
    parser.add_argument('--num', type=int, default=-1, help="The number of files to combine")
    parser.add_argument('--prefix', type=str, default='sft', help="The prefix of the files to combine")
    parser.add_argument('--only_iter', action='store_true')
    # parser.add_argument("--new_path", type=str, help="The path of the new json file")
    args = parser.parse_args()
    combine_json_data(args.data_path, args.save_path, prefix=args.prefix, num=args.num,
                      only_iter=args.only_iter)