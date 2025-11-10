import pathlib
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dataclasses import dataclass, field
from datasets import Dataset
from Evol_Instruct.MCTS.utils import extract_template
from Evol_Instruct.training.value_train import print_on_main
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl import ScriptArguments
# from trl import SFTScriptArguments as ScriptArguments
from Evol_Instruct import client
from Evol_Instruct.utils.utils import add_proxy, proxy_decorator
from peft import PeftModel
from tqdm import tqdm
import torch

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
@dataclass
class SFTScriptArguments(ScriptArguments):
    dataset_name: str = field(metadata={"help": "The name of the dataset to use."}, default=None)
    data_path: str = field(metadata={"help": "Path to the data file."}, default=None)
    test_split_ratio: float = field(metadata={"help": "Ratio of the test split."}, default=0.1)
    
    # if load a lora-tuned sft model
    tuned_lora_path: list[str] = field(metadata={"nargs": "+", "help": "Path to the lora-tuned model."}, default_factory=list)
    
    learn_advantage: bool = field(metadata={"help": "only learn the trace where the next step possesses higher value than last step."}, default=False)
    filter_extreme: int = field(metadata={"help": "learn how many trajectories for samples with no negative trajs"}, default=-1)
    remain_highest: int = field(metadata={"help": "keep how many highest value responses"}, default=-1)
    
    
def return_prompt_and_responses(samples):
    new_list = []
    for sample in samples:
        conversations = sample['conversations']
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i, conv in enumerate(conversations):
            if i % 2 == 0:
                # the user
                message = {"role": "user", "content": conv['value']} 
            else:
                # the gpt
                message = {"role": "assistant", "content": conv['value']}
            messages.append(message)
                
        new_list.append({"messages": messages})
    return new_list

def input_format(question, response, tokenizer):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": response})
    total_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return total_input

def format_tree_data(samples, data_usage, learn_advantage, tokenizer):
    new_list = []
    for sample in tqdm(samples, ncols=60):
        trajectory_w_value = sample["pos"]
        for trajectory in trajectory_w_value:
            trajectory = trajectory[0]
            # if trajectory[-1] != 1 and trajectory[-1][2] != 1:
            #     continue

            response_ids = []
            label_ids = []
            response = ""
            step_idx = 0
            for traj_idx in range(len(trajectory)):
                traj = trajectory[traj_idx][0]
                traj_str = f"Step {traj_idx}: " + traj + ("\n\n" if traj_idx != len(trajectory)-1 else "")
                # traj_str = "".join([f"""{PROMPT_TOKENS["cot_step"]["begin"].format(step_id=step_idx+state_id)}{traj_statae}{PROMPT_TOKENS["cot_step"]["end"]}""" for state_id, traj_statae in enumerate(traj)])
                step_idx = 0

                if traj_idx == 0 or trajectory[traj_idx][1] >= trajectory[traj_idx-1][1] or (not learn_advantage):
                    learn_flag = True
                else:
                    learn_flag = False
                
                response += traj_str
                response_id = tokenizer.encode(traj_str, add_special_tokens=False)
                response_ids += response_id
                if learn_flag == True:
                    label_ids += response_id
                else:
                    label_ids += [-100] * len(response_id)

            total_input_str = input_format(sample["question"], response, tokenizer)
            example = tokenizer(total_input_str, add_special_tokens=False)
            response_begin_index = ",".join(map(str, example["input_ids"])).index(",".join(map(str, response_ids)))
            response_begin_index = ",".join(map(str, example["input_ids"]))[:response_begin_index].count(",")
            example["labels"] = example["input_ids"][:]
            example["labels"][response_begin_index:response_begin_index+len(response_ids)] = label_ids
            
            new_list.append(example)
            if data_usage == "sample":
                break

    return new_list

def obtain_dataset(data_path, learn_advantage, script_args):
    data = client.read(data_path)
    if 'mcts' in data_path:
        new_data = []
        # data_list = format_tree_data(data, 'all', learn_advantage, tokenizer)
        for item in data:
            category = 'pos'
            # if only_max_value:
            responses = []
            
            for term in item[category]:
                if isinstance(term[0], str):
                    response = term[0]
                    value = 1
                else:
                    response = "\n\n".join([f"Step {i}: " + step[0] for i, step in enumerate(term[0])])
                    value = min([step[1] for step in term[0][1:]])

                
                responses.append((response, value))
            # if keep_largest_value != -1:
            #     responses = sorted(responses, key=lambda x: x[1], reverse=True)[:keep_largest_value]
            #     responses = [response for response, value in responses]
            # else:
            if script_args.filter_extreme != -1:
                if len(item['neg']) == 0:
                    # select the highest `script_args.filter_extreme` value responses
                    responses = sorted(responses, key=lambda x: x[1], reverse=True)[:script_args.filter_extreme]
                    if not responses:
                        continue
            if script_args.remain_highest != -1:
                responses = sorted(responses, key=lambda x: x[1], reverse=True)[:script_args.remain_highest]
            responses = [response for response, value in responses]

                
            for response in responses:
                new_data.append({
                    "id": item['id'],
                    "conversations": [
                        {"from": "human", "value": item['question']},
                        {"from": "gpt", "value": response}
                    ],
                    "label": 1.0,
                    "inter": True if term == 'inter' else False
                })
        data_list = return_prompt_and_responses(new_data)
    elif "o3-mini" in data_path:
        new_data = []
        wrong_cnt = 0
        for sample in data: 
            answer_idx = sample['eval']['answer_idx']
            answer = sample['eval']['answer']
            for response in sample['gpt_responses']:
                response = response.strip()
                predict_answer = extract_template(response, r"(?:[a-z]+ )?answer")

                if predict_answer is None:
                    predict_answer = extract_template(response, 'correct answer') 
                if predict_answer is None:
                    predict_answer = extract_template(response, 'final answer') 
                if predict_answer is None:
                    save = False
                else:
                    predict_answer = predict_answer.strip().rstrip(".").strip("{").strip("}").split("answer")[-1]
                    predict_answer = predict_answer.strip(":").replace("}}", "").strip().strip(".").strip('"').strip("'")
                    save = False
                    
                    if len(predict_answer) == 1 and predict_answer == answer_idx:
                        save = True
                    if re.match( r'^[A-D]\.\s.*', predict_answer) and predict_answer[0] == answer_idx:
                        save = True
                    elif len(predict_answer) != 1 and all(each.lower() in [x.lower() for x in predict_answer.split(" ")] for each in answer.split(" ")):
                        save = True
                    elif len(predict_answer) != 1 and predict_answer.lower() in answer.lower():
                        save = True
                if save:
                    each_line_split = response.split("\n\n")
                    new_response = "\n\n".join(each_line_split[:-1]) + f"\n\nThe answer is {answer_idx}."
                    new_data.append({
                        "id": sample['id'],
                        "conversations": [
                            {"from": "human", "value": sample['input']},
                            {"from": "gpt", "value": new_response}
                        ],
                    })
                else:
                    wrong_cnt += 1
                
        print_on_main(f"There are {wrong_cnt} wrong responses generated by o3-mini")
          
        data_list = return_prompt_and_responses(new_data)
    else:
        data_list = return_prompt_and_responses(data)
    dataset = Dataset.from_list(data_list)
    return dataset 

def split_dataset(dataset: Dataset, test_size=0.1):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split


def dynamic_collate_fn(batch):
    # 获取当前 batch 的 input_ids 和 labels
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch] if "attention_mask" in batch[0] else None
    labels = [item["labels"] for item in batch]
    
    # 找到当前 batch 中的最大长度
    max_length = max(len(ids) for ids in input_ids)
    
    # 动态 padding input_ids 和 attention_mask
    padded_input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids])
    padded_label = torch.tensor([ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in labels])
    if attention_masks:
        padded_attention_mask = torch.tensor([mask + [0] * (max_length - len(mask)) for mask in attention_masks])
    else:
        padded_attention_mask = (padded_input_ids != tokenizer.pad_token_id).long()  # 动态生成 mask
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_label
    }

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    # quantization_config = get_quantization_config(model_config)
    # model_kwargs = dict(
    #     revision=model_config.model_revision,
    #     trust_remote_code=model_config.trust_remote_code,
    #     attn_implementation=model_config.attn_implementation,
    #     torch_dtype=model_config.torch_dtype,
    #     use_cache=False if training_args.gradient_checkpointing else True,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    if hasattr(config, 'quantization_config'):
        config.quantization_config["use_exllama"] = False
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        config=config,
        trust_remote_code=model_config.trust_remote_code,
        # use_fast=True,
    )
    print_on_main(f"Loading {model_config.model_name_or_path} over...")
    model_dtype = model.dtype
    print_on_main(script_args.tuned_lora_path)
    if script_args.tuned_lora_path is not None and script_args.tuned_lora_path != ["None"]:
        for lora_path in script_args.tuned_lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print_on_main(f"Load lora weights from {lora_path}")
        model = model.to(model_dtype)
    # training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True, model_max_length=training_args.max_seq_length
    )
    if tokenizer.pad_token_id is None:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = "<|finetune_right_pad_id|>" if "3.1-8B" in model_config.model_name_or_path else tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = obtain_dataset(script_args.data_path, script_args.learn_advantage, script_args)
    if training_args.eval_strategy == 'no':
        dataset = {"train": dataset}
    else:
        dataset = split_dataset(dataset, test_size=script_args.test_split_ratio)
    print_on_main(model_config)
    for key in dataset:
        dataset[key] = dataset[key].filter(
        lambda x: len(tokenizer.apply_chat_template(x['messages'], tokenize=True)) <= 4096
    )

    ################
    # Training
    ################
    with add_proxy():
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_config),
            # data_collator=dynamic_collate_fn
        )
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)