import argparse

from tqdm import trange 
from Evol_Instruct.models.modeling_value_llama import ValueModel
from Evol_Instruct.solver.sc_vm_solver import SCVMSolver
from Evol_Instruct import client
from transformers import AutoTokenizer
import torch 
import json
from Evol_Instruct.utils.utils import get_chunk

torch.set_default_device("cuda")
class Server:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate the dataset with a specific method.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', nargs='+', type=str, default=None)
    parser.add_argument('--model_type', default='prm', type=str, help='Model type to use for evaluation.')  
    parser.add_argument('--infer_rule', type=str, default='prm-min-max')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--num_chunks', type=int, default=1, help='Number of chunks to split the dataset into.')
    parser.add_argument('--chunk_idx', type=int, default=0, help='Chunk ID to process.')
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = args.data_path
    print(args)
    
    if args.model_base is None or args.model_base == "None":
        tokenizer_path = args.model_path[0]
    else:
        tokenizer_path = args.model_base 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    value_model = ValueModel(args.model_base, args.model_path, args.model_type)
    print("Initialize value model over...")
    server = Server(tokenizer)

    new_items = []
    
    
    items = client.read(args.data_path)
    items = get_chunk(items, args.num_chunks, args.chunk_idx)
    if client.exists(args.save_path):
        # resume from the save_path
        already_process_data = client.read(args.save_path)
        processed_ids = set([item["id"] for item in already_process_data])
        items = [item for item in items if item["id"] not in processed_ids]
        # fp.close()
        fp = open(args.save_path, 'a')
        print(f"Already processed {len(already_process_data)} items, remaining {len(items)} items to process.")
    else:
        fp = open(args.save_path, 'w')
    solver = SCVMSolver(server, fp, value_model=value_model,
                        infer_rule=args.infer_rule,)
        
    for i in trange(0, len(items), args.batch_size):
        batch_items = items[i:i + args.batch_size]
        # print(batch_items)
        # batch_alpaca_items = 
        batch_new_items = solver.rescore(batch_items, value_model_type=args.model_type)
        for each_item in batch_new_items:
            fp.write(json.dumps(each_item, ensure_ascii=False) + "\n")

        new_items.extend(batch_new_items)
    

    fp.close()