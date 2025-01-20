# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Full training:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048

LoRA:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward-LoRA \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

from collections import defaultdict
import warnings
from dataclasses import dataclass, field
from typing import Optional, Callable, Union, Any
import torch
import pathlib
import warnings
from datasets import load_dataset, Dataset, DatasetDict
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, AutoModel
from transformers import Trainer
from accelerate import Accelerator
 
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
import time
# from Evol_Instruct.training.dpo_train import rank0_print
from Evol_Instruct import client
from Evol_Instruct.utils.utils import add_proxy
from Evol_Instruct.models.model_rm import set_pairrm_tokenizer, set_ultra_rm_tokenizer, pairrm_prompt_template
import torch.distributed as dist


def print_on_main(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)




# 获取当前时间的精确秒数作为种子
seed = int((time.time() * 1000) % (2 ** 32))

# 用于生成一个伪随机数的函数
def pseudo_random(seed):
    seed ^= seed >> 21
    seed ^= seed << 35
    seed ^= seed >> 4
    seed *= 2685821657736338717
    seed ^= seed >> 21
    seed ^= seed << 35
    seed ^= seed >> 4
    return (seed & ((1 << 32) - 1)) / (2 ** 32)

# 打乱列表
def shuffle_list(lst):
    n = len(lst)
    for i in range(n - 1, 0, -1):
        j = int(pseudo_random(seed) * (i + 1))  # 生成一个随机索引
        lst[i], lst[j] = lst[j], lst[i]  # 交换元素



@dataclass
class RewardScriptArguments:
    dataset_name: str = field(metadata={"help": "Name of the dataset to use."}, default=None)
    data_path: str = field(metadata={"help": "Path to the data file."}, default=None)
    test_split_ratio: float = field(metadata={"help": "Ratio of the test split."}, default=0.1)
    rm_name: str = field(default='llama3')
    # if load a lora-tuned sft model
    tuned_lora_path: list[str] = field(metadata={"nargs": "+", "help": "Path to the lora-tuned model."}, default_factory=list)
    
    
    learn_rollout_value: bool = field(default=False, metadata={"help": "Whether to learn the rollout value."})
    learn_orm: bool = field(default=False, metadata={"help": "Whether to learn the ORM."})

    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False

def obtain_dataset_each(data, learn_rollout_value=False, learn_orm=False, train_pair_per_instance=-1):
    new_data = []
    for item in data:
        # cur_item_data = []
        step_wise_data = defaultdict(list)
        cmp_group = ['pos', 'neg'] if learn_orm else ['pos', 'neg', 'inter']
        for category in cmp_group:
            for term in item[category]:
                if isinstance(term[0], str):
                    response = term[0]
                    num_step = len(response.split("\n\nStep"))
                else:
                    num_step = len(term[0])
                    response = "\n\n".join([f"Step {i}: " + step[0] for i, step in enumerate(term[0])])
                cur_item = {
                    "id": item['id'],
                    "question": item['question'],
                    "response": response,
                    "label": term[2] if learn_rollout_value else term[1],
                    "inter": True if category == 'inter' else False,
                    "num_step": num_step
                }
                step_wise_data[num_step].append(cur_item)
        # group the data by the num_step field
        # step_wise_data = {}
        # for each step's list, sort the data by the label value and remaining items with different label
        if learn_orm:
            overall_data = []
            for num_step, step_data in step_wise_data.items():
                overall_data.extend(step_data)
            step_wise_data = {1: overall_data}
        
        for num_step, step_data in step_wise_data.items():
            step_data.sort(key=lambda x: x['label'], reverse=True)
            possible_pairs = []
            for i in range(len(step_data)):
                for j in range(i + 1, len(step_data)):
                    if step_data[j]['label'] < step_data[i]['label']:
                        possible_pairs.append((i, j))
            if train_pair_per_instance != -1:
                shuffle_list(possible_pairs)
                possible_pairs = possible_pairs[:train_pair_per_instance]
            for i, j in possible_pairs:
                new_data.append(
                    {
                        "chosen": [{"role": "user", "content": step_data[i]['question']}, {"role": "assistant", "content": step_data[i]['response']}],
                        "rejected": [{"role": "user", "content": step_data[j]['question']}, {"role": "assistant", "content": step_data[j]['response']}],
                        "margin": step_data[i]['label'] - step_data[j]['label']
                    }
                )
            

    shuffle_list(new_data)
    dataset = Dataset.from_list(new_data)
    return dataset


def process_pair_dataset(batch, tokenizer):
    new_examples = {
        "left_input_ids": [],
        "left_attention_mask": [],
        "right_input_ids": [],
        "right_attention_mask": [],
    }
    for chosen, rejected in zip(batch['chosen'], batch['rejected']):
        context = tokenizer.apply_chat_template(chosen[:-1], tokenize=False)
        left_prompt = pairrm_prompt_template.format(context=context,
                                                    response_A=chosen[-1]['content'],
                                                    response_B=rejected[-1]['content'])
        right_prompt = pairrm_prompt_template.format(context=context,
                                                        response_A=rejected[-1]['content'],
                                                        response_B=chosen[-1]['content'])
        left_message = [
            {"role": "user", "content": left_prompt}
        ]
        right_message = [
            {"role": "user", "content": right_prompt}
        ]
        left_input_ids = tokenizer(tokenizer.apply_chat_template(message, tokenize=False).replace(tokenizer.bos_token, ""), return_tensors='pt')
        right_input_ids = tokenizer(tokenizer.apply_chat_template(message, tokenize=False).replace(tokenizer.bos_token, ""), return_tensors='pt')
        new_examples['left_input_ids'].append(left_input_ids['input_ids'])
        new_examples['left_attention_mask'].append(left_input_ids['attention_mask'])
        new_examples['right_input_ids'].append(right_input_ids['input_ids'])
        new_examples['right_attention_mask'].append(right_input_ids['attention_mask'])
    return new_examples

def compute_pair_accuracy(eval_pred):
    predictions, labels = eval_pred 
    equal_num = (predictions == 0.5).sum()
    if equal_num > 0:
        warnings.warn(
            f"There are {equal_num} out of {len(predictions)} instances where the predictions "
            "for both options are equal. These instances are ignored in the accuracy computation.",
            UserWarning,
        )
    accuracy = (predictions > 0.5).sum() / int((predictions != 0.5).sum())
    return {"accuracy": accuracy}
    

class EvaluateTrainer(Trainer):
    def __init__(self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[dict] = None,):
        
        
        self.token_id_A = processing_class.encode("A", add_special_tokens=False)
        self.token_id_B = processing_class.encode("B", add_special_tokens=False)
        assert len(self.token_id_A) == 1 and len(self.token_id_B) == 1
        self.token_id_A = self.token_id_A[0]
        self.token_id_B = self.token_id_B[0]
        self.temperature = 1.0
        
        if compute_metrics is None:
            self.compute_metrics = compute_pair_accuracy
        
        fn_kwargs = {"tokenizer": self.processing_class}
        eval_dataset = eval_dataset.map(
            process_pair_dataset,
            batched=True,
            fn_kwargs=fn_kwargs,
            num_proc=args.dataset_num_proc
        )
        eval_dataset = eval_dataset.filter(
                        lambda x: len(x["left_input_ids"]) <= max_length,
                        num_proc=args.dataset_num_proc,
                    )
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        left_outputs = model(input_ids=inputs['left_input_ids'], attention_mask=inputs['left_attention_mask'], return_dict=True)['logits']
        right_outputs = model(input_ids=inputs['right_input_ids'], attention_mask=inputs['right_attention_mask'], return_dict=True)['logits']
        return None, {
            "left_logits": left_outputs,
            "right_logits": right_outputs
        }
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys = None
    ):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            _, logits = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (None, None, None)
        
        def compute_one_side_prob(logit_A, logit_B, chosen_position):
            Z = torch.exp(logit_A / self.temperature) +torch.exp(logit_B / self.temperature)
            logit_chosen = logit_A if chosen_position == 0 else logit_B 
            prob_chosen = torch.exp(logit_chosen / self.temperature) / Z
            return prob_chosen 
        
        left_logit_A = logits['left_logits'][:, -1, self.token_id_A]
        left_logit_B = logits['left_logits'][:, -1, self.token_id_B]
        right_logit_A = logits['right_logits'][:, -1, self.token_id_A]
        right_logit_B = logits['right_logits'][:, -1, self.token_id_B]
        
        left_prob_chosen = compute_one_side_prob(left_logit_A, left_logit_B, 0)
        right_prob_chosen = compute_one_side_prob(right_logit_A, right_logit_B, 1)
        
        avg_prob_chosen = (left_prob_chosen + right_prob_chosen) / 2
        
        # loss = loss.detach()
        # logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        # logits = nested_detach(logits)
        # # Stack accepted against rejected, mean over logits
        # # and softmax to get preferences between accepted and rejected to sum to 1
        # logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return None, avg_prob_chosen, labels


def uniform_sample(lst, num):
    if len(lst) <= num:
        return lst
    step = len(lst) // num
    sampled_data = []
    for index in range(0, len(lst), step):
        sampled_data.append(lst[index])
    return sampled_data




def obtain_dataset(data_path, learn_rollout_value=False, learn_orm=False,):
    total_data = client.read(data_path)
    eval_dataset = obtain_dataset_each(total_data, learn_rollout_value, learn_orm, train_pair_per_instance=-1)
    # total_data = client.read(data_path)
    # dataset = obtain_dataset_each(sampled_data, learn_rollout_value, learn_orm)
    # return 
    
        
    return eval_dataset




def split_dataset(dataset: Dataset, test_size=0.1):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split

if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    print_on_main(script_args)
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    model_args.lora_task_type = "SEQ_CLS"
    
    
    ##############
    # Load dataset
    ##############
    dataset = obtain_dataset(script_args.data_path, learn_rollout_value=script_args.learn_rollout_value, learn_orm=script_args.learn_orm)
    
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True, model_max_length=training_args.max_length
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>" if "3.1-8B" in model_args.model_name_or_path else tokenizer.eos_token
    trainer_cls = RewardTrainer
    if script_args.rm_name != 'llama3':
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True, **model_kwargs
        )
        
        if script_args.rm_name == 'ultra_rm':
            tokenizer = set_ultra_rm_tokenizer(tokenizer)
        elif script_args.rm_name == 'pair_rm':
            tokenizer = set_pairrm_tokenizer(tokenizer)
            trainer_cls = EvaluateTrainer
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
        
    if script_args.tuned_lora_path is not None and script_args.tuned_lora_path != ['None']:
        for lora_path in script_args.tuned_lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print_on_main(f"Load lora weights from {lora_path}")
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id
    print_on_main(model)
    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    
    # dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    # test_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")
    # dataset = {"train": dataset, "test": test_dataset}
    # dataset = split_dataset(dataset, script_args.test_split_ratio)
    

    ##########
    # Training
    ##########
    with add_proxy():
        
        trainer = trainer_cls(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            peft_config=get_peft_config(model_args),
        )


        ############################
        # Save model and push to Hub
        ############################
        # trainer.save_model(training_args.output_dir)

        if training_args.eval_strategy != "no":
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Save and push to hub
