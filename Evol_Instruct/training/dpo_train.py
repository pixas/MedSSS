from pyexpat import model
from datasets import Dataset
from Evol_Instruct import client, logger
import pathlib
from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from functools import partial

from Evol_Instruct.utils.utils import add_proxy

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
def return_prompt_and_responses(samples, tokenizer):
    # print(list(samples.keys()))
    return {
        "prompt": [tokenizer.apply_chat_template(sample,
                                                 tokenize=False,
                                                 add_generation_prompt=True) for sample in samples['prompt']],
        "chosen": [sample[0]['content'] for sample in samples['chosen']],
        "rejected": [sample[0]['content'] for sample in samples['rejected']]
    }

def obtain_dataset(data_path, tokenizer):
    # rank0_print(f"Train data path: {data_path}")
    # print("Train data path: ", data_path)
    data = client.read(data_path)
    # print(data)
    dataset = Dataset.from_list(data)
    original_columns = dataset.column_names
    process_func = partial(return_prompt_and_responses, tokenizer=tokenizer)
    dataset = dataset.map(process_func, batched=True, remove_columns=original_columns)
    return dataset

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer    


def train_dpo(config, model, tokenizer, dataset):
    trainer = DPOTrainer(config, model, tokenizer)
    trainer.train(dataset)
    return trainer

def split_dataset(dataset: Dataset, test_size=0.1):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split



# ModelConfig
# 
# class ModelConfiguration(ModelConfig):
@dataclass
class DPOScriptArguments(ScriptArguments):
    data_path: str = field(metadata={"help": "Path to the data file."}, default=None)
    test_split_ratio: float = field(metadata={"help": "Ratio of the test split."}, default=0.1)
    
    # if load a lora-tuned sft model
    tuned_lora_path: str = field(metadata={"help": "Path to the lora-tuned model."}, default=None)

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    if script_args.tuned_lora_path:
        model = PeftModel.from_pretrained(model, script_args.tuned_lora_path)
        model = model.merge_and_unload()
        rank0_print("Merging pre-trained SFT Lora params")
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name)
    dataset = obtain_dataset(script_args.data_path, tokenizer)
    dataset = split_dataset(dataset, script_args.test_split_ratio)
    assert script_args.dataset_train_split in dataset, f"{script_args.dataset_train_split} not in dataset"
    ##########
    # Training
    ################
    with add_proxy():
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            # processing_class=tokenizer,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_length=training_args.max_length,
        )
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        # trainer.train()
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)