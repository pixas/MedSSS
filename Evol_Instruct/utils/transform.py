import json 
import os
import sys 
# print(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from Evol_Instruct.utils.utils import client 
import argparse
def transform_jsonl2json(data_path, new_path):
    data = client.read(data_path)
    # new_name = data_path.split("/")[-1].split(".jsonl")[0] + ".json"
    # new_path = os.path.join(new_path, new_name)
    client.write(data, new_path)
    os.remove(data_path)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform jsonl to json")
    parser.add_argument("--data_path", type=str, help="The path of the jsonl file")
    parser.add_argument("--new_path", type=str, help="The path of the new json file")
    args = parser.parse_args()
    transform_jsonl2json(args.data_path, args.new_path)