import os 
import json
import pdb
from Evol_Instruct.utils.utils import client 
from transformers import AutoTokenizer, AutoModelForCausalLM

from collections import defaultdict
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
# from Evol_Instruct.models.vllm_support import vllm_clean_generate, get_vllm_model

torch.set_default_device("cuda")


def get_data_path(name):
    path = os.path.join("/mnt/petrelfs/jiangshuyang.p/oss/datasets/medical_train", name + ".json")
    return path 

def count_vocab(name):
    path = get_data_path(name)
    data = client.read(path)
    
    vocab = defaultdict(int)
    for item in data:
        conv = item['conversations']
        new_conv = [{"role": "system", "content": "You are a helpful assistant."}] + [{
            "role": "user" if turn['from'] == 'human' else "assistant",
            "content": turn['value'] 
        } for turn in conv]
        
        text = tokenizer.apply_chat_template(new_conv, tokenize=False,
                                add_generation_prompt=True)
        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0].tolist()
        # tokens_count = defaultdict(int)
        for token in tokens:
            vocab[token] += 1
    return vocab

        

def get_loss(logits, labels, attention_mask, vocab_size):
    from torch.nn import CrossEntropyLoss
    labels = labels.masked_fill(~attention_mask, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    B, N, C = shift_logits.shape
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    # this loss is [-1, ], we need to reshape it to [B, N]
    loss = loss.reshape(B, N)
    token_loss = loss.clone()
    # we must know that some positions are 0-loss because of ignore_index, we need to ignore these
    loss_sum = loss.sum(dim=-1)
    loss_actual_position = torch.not_equal(loss, 0).sum(dim=-1)
    loss = loss_sum / loss_actual_position  # [B, ]
    return loss, token_loss


def obtain_nll_dist(model, tokenizer, name, batch_size=4):
    # compute nll loss and plot the nll loss distribution
    path = get_data_path(name)
    data = client.read(path)
    all_nll_loss = []
    batch_idx = 0
    batch_data = []
    tokenizer.padding_side = "left"

    for item in tqdm(data, total=len(data)):
        conv = item['conversations']
        new_conv = [{"role": "system", "content": "You are a helpful assistant."}] + [{
            "role": "user" if turn['from'] == 'human' else "assistant",
            "content": turn['value'] 
        } for turn in conv]
        
        text = tokenizer.apply_chat_template(new_conv, tokenize=False,
                                add_generation_prompt=True)
        if batch_idx < batch_size:
            batch_data.append(text)
            batch_idx += 1

        if batch_idx == batch_size or item == data[-1]:

            tokens = tokenizer(batch_data, return_tensors='pt', padding=True)['input_ids']
            input_ids = tokens.to(device='cuda', non_blocking=True)
            with torch.inference_mode():
                # pdb.set_trace()
                # print(tokens.shape)
                try:
                    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
                    outputs = model(input_ids, labels=input_ids)
                    logits = outputs.logits
                    loss, token_loss = get_loss(logits, input_ids, attention_mask, model.config.vocab_size)
                except torch.OutOfMemoryError:
                    loss_half = []
                    size = input_ids.shape[0] // 2
                    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
                    for i in range(0, input_ids.shape[0], input_ids.shape[0] // 2):
                        input_ids_half = input_ids[i:i+size]
                        attention_mask_half = attention_mask[i:i+size]
                        outputs = model(input_ids_half, labels=input_ids_half)
                        logits = outputs.logits
                        loss_half.append(get_loss(logits, input_ids_half, attention_mask_half, model.config.vocab_size)[0])
                    loss = torch.cat(loss_half, dim=0)
                # loss = outputs.loss

            all_nll_loss.extend(loss.cpu().numpy().tolist())
            # if isinstance(loss.item(), list):
            #     all_nll_loss.extend(loss.item())
            # else:
            #     all_nll_loss.append(loss.item())
            batch_idx = 0
            batch_data = []
    # obtain the dist of nll loss
    assert len(all_nll_loss) == len(data)
    return all_nll_loss

def plot_dist(data_dict, **kwargs):
    sns.set_style("whitegrid")  # 设置绘图风格
    # del data_dict['GPT-4o-mini']
    # print(len(data_dict))
    # sns.histplot(dist, kde=True, bins=30, stat='density')
    palette = sns.color_palette('rocket', len(data_dict))
    for (file_name, dist), color in zip(data_dict.items(), palette):
        print(file_name, dist.shape)
        sns.kdeplot(dist, 
                fill=False, label=file_name, color=color,
   alpha=.5, clip=(0, 5))

    # 添加标题和标签
    # plt.title('Distribution of MMed EN seed')
    plt.xlabel(kwargs.pop("xlabel", 'Value'))
    plt.ylabel(kwargs.pop("ylabel", "Value"))

    # 显示图例
    plt.legend(title='Dataset')

    # 显示图形
    # plt.show()
    plt.savefig(kwargs.pop("save_path", "images/dist.pdf"), dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_idx', type=int, default=0)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--plot_dist', action="store_true")
    # parser.add_argument('--sa')
    args = parser.parse_args()
    
    data_name = ['mmed_en_train', 'gpt-4o-mini_mmeden_evol-all', 'gpt-4o-mini-taiyi_mcqa_10000_mmeden_mix', 'gpt-4o-mmed_en_taiyi10000_evol_mix', 'gpt-4o-mini_taiyi_mcqa_10000_evol-random', 'gpt-4o-mini_pubmedqa_evol_random']

    print(f"Processing {args.data_name}")
    model_path = "/mnt/petrelfs/jiangshuyang.p/models/Meta-Llama-3-8B-Instruct"
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if args.plot_dist:
        # data = np.load(args.save_path)
        subdir = os.path.dirname(args.save_path)
        # obtain all .npy files
        all_file_name = os.listdir(subdir)
        file_name_mapping = {}
        for file_name in all_file_name:
            if "gpt-4o" in file_name and 'all' not in file_name:
                file_name_mapping[file_name] = "GPT-4o-mini"
            elif "llama3" in file_name:
                file_name_mapping[file_name] = "Llama3.1 70B"
            elif "qwen2" in file_name:
                file_name_mapping[file_name] = "Qwen2 72B"
            elif "gpt-4o" in file_name and "all" in file_name:
                file_name_mapping[file_name] = "GPT-4o-mini-all"
            else:
                file_name_mapping[file_name] = "Seed"
                
        # all_data = [os.path.join(subdir, x) for x in os.listdir(subdir)]
        data_dict = {file_name_mapping[file_name]: np.load(os.path.join(subdir, file_name)) for file_name in all_file_name if file_name in file_name_mapping}
        # print(data_dict['GPT-4o-mini'].shape, data_dict['GPT-4o-mini-all'].shape)
        figure_save_path = f"images/all_data_test.pdf"
        plot_dist(data_dict, xlabel='NLL Loss', ylabel='Density', save_path=figure_save_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        assert tokenizer.pad_token_id is not None
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        
        dist = obtain_nll_dist(model, tokenizer, args.data_name, batch_size=args.batch_size)
        print("Process over")
    # sns.set_style("whitegrid")  # 设置绘图风格
    # sns.histplot(dist, kde=True, hist=True, bins=30, color='b', 
    #             hist_kws={"edgecolor": 'black'},
    #             kde_kws={"color": "r", "lw": 3, "label": "KDE"})

    # # 添加标题和标签
    # plt.title('Distribution of the Data')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # # 显示图例
    # plt.legend()

    # # 显示图形
    # plt.show()
        dist_data = np.array(dist)
        os.remove(args.save_path)
        np.save(args.save_path, dist_data)
        