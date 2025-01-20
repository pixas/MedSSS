from tqdm import tqdm
from Evol_Instruct import client
from Evol_Instruct.models.modeling_value_llama import LlamaForValueFunction 
import argparse 
from transformers import AutoTokenizer
import torch 
from peft import PeftModel



def score_item(item, tokenizer, value_function):
    # except for the neg and pos, use value_function to score the inter instances
    inter_instances = item['inter']
    problem = item['question']
    conversations = [[
        {"role": "user", "content": problem},
        {"role": "assistant", "content": trajectory}
    ] for trajectory, _ in inter_instances]
    texts = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(texts, return_tensors='pt', padding=True)
    tokens = {k: v.to(value_function.device) for k, v in tokens.items()}
    with torch.inference_mode():
        values = value_function(**tokens)
    value_model_scores = values[0]
    for instance, value_model_scores in zip(inter_instances, value_model_scores):
        instance[1] = 0.5 * (value_model_scores.item() + instance[1])
        # instance[1] = value_model_scores.item()
    # for instance in inter_instances:
    #     trajectory = instance[0]
    #     conversation = [
    #         {"role": "user", "content": problem},
    #         {"role": "assistant", "content": trajectory}
    #     ]
    #     text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    #     tokens = tokenizer(text, return_tensors='pt', padding=True)
    #     tokens = {k: v.to(value_function.device) for k, v in tokens.items()}
    #     with torch.inference_mode():
    #         values = value_function(**tokens)
    #     value_model_scores = values[0].item()
    #     instance[1] = value_model_scores

def score_data(args):
    torch.set_default_device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model_base = LlamaForValueFunction.from_pretrained(
        args.model_base,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    value_model = PeftModel.from_pretrained(model_base, args.model_path)
    value_model = value_model.merge_and_unload().to(torch.float16)
    print("Load value model over")
    data = client.read(args.data_path)
    
    for item in tqdm(data, total=len(data)):
        score_item(item, tokenizer, value_model)
    client.write(data, args.output_path, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    score_data(args)