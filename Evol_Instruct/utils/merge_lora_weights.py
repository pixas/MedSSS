from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import torch 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str)
    parser.add_argument("--model_path", type=str, nargs="+")
    
    parser.add_argument('--save_path', type=str)
    
    args = parser.parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    for each_path in args.model_path:
        model = PeftModel.from_pretrained(model, each_path)
        model = model.merge_and_unload()
        # model
    # model = PeftModel.from_pretrained(model_base, args.model_path)
    model = model.to(torch.float16)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    
    
    