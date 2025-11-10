# import torch
from transformers.models.llama.modeling_llama import LlamaForSequenceClassification, LLAMA_INPUTS_DOCSTRING, SequenceClassifierOutputWithPast, add_start_docstrings_to_model_forward
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForTokenClassification
from transformers import PreTrainedModel, Trainer
from trl import PRMTrainer
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn 
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple
import torch
# from typing import Optional, Union, List, Tuple
# from transformers.cache_utils import Cache
# from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
# from transformers.models.qwen2.modeling_qwen2 import Qwen2ForSequenceClassification, QWEN2_INPUTS_DOCSTRING


@dataclass
class ProcessTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    penalty_logits: torch.FloatTensor=None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class ProcessRewardModel(Qwen2ForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # print(self.loss_function)
        # print(self.config)
        self.config.loss_type = "ForTokenClassification"
        # self.dropout_2 = nn.Dropout(classifier_dropout)
        # self.backbone = backbone  # 传入预训练的PLM
        # self.mc_head = nn.Linear(config.hidden_size, 1)  # MC值回归头
        self.penalty_head = nn.Linear(config.hidden_size, self.num_labels)  # 单调性判别头
        
    def forward(self, input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)
        penalty_logits = self.penalty_head(sequence_output)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)
        
        # 这里的outputs是一个TokenClassifierOutput对象
        if not return_dict:
            output = (logits, penalty_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ProcessTokenClassifierOutput(
            loss=loss,
            logits=logits,
            penalty_logits=penalty_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


        return {**outputs, "penalty_logits": penalty_logits}

# class ProcessRewardModelWrapper:
#     def __init__(self, token_cls_model):
#         self.token_cls_model = token_cls_model 
#         self.num_labels = token_cls_model.num_labels 
#         self.penalty_head = nn.Linear(token_cls_model.score.in_features, self.num_labels)  # 单调性判别头
        
    
#     def forward(self, input_ids, attention_mask=None, penalties=None, **kwargs):
#         outputs = self.token_cls_model(input_ids, attention_mask=attention_mask)
#         last_hidden = outputs.hidden_states
#         # mc_logits = torch.sigmoid(self.mc_head(last_hidden)).squeeze(-1)  # [batch,]
#         penalty_logits = self.penalty_head(last_hidden)

#         return {**outputs, "penalty_logits": penalty_logits}

import torch
@dataclass
class ProcessDataCollator(DataCollatorForTokenClassification):
    def __call__(
        self, features
    ) -> Dict[str, torch.Tensor]:
        # 先调用父类处理padding
        # print(features[0]['penalties'])
        penalty_name = "penalties"
        penalties = [feature[penalty_name] for feature in features] if penalty_name in features[0].keys() else None

        no_penalty_features = [{k: v for k, v in feature.items() if k != penalty_name} for feature in features]
        batch = super().__call__(no_penalty_features)
        if penalties is None:
            return batch
        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        # 添加自定义字段
        # print(list(batch.keys()))
        padding_side = self.tokenizer.padding_side
        sequence_length = batch['input_ids'].shape[1]
        if padding_side == "right":
            batch[penalty_name] = [
                to_list(penalty) + [self.label_pad_token_id] * (sequence_length - len(penalty)) for penalty in penalties
            ]
        else:
            batch[penalty_name] = [
                [self.label_pad_token_id] * (sequence_length - len(penalty)) + to_list(penalty) for penalty in penalties
            ]

        batch[penalty_name] = torch.tensor(batch[penalty_name], dtype=torch.int64)
    
        return batch
class ProcessTrainer(PRMTrainer):
    def __init__(self, *args, alpha=0.3, lambda_penalty=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # 控制penalty损失权重
        self.lambda_penalty = lambda_penalty  # 推理时的惩罚系数
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # print(list(inputs.keys()))
        penalty = inputs.pop("penalties")  
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # print(list(outputs.keys()))
        penalty_logits = outputs['penalty_logits']
        
        mc_loss = loss
        # mc_loss = F.mse_loss(mc_logits, mc_labels)  # 回归任务
        
        # 4. 计算单调性惩罚损失
        # is_worse = (mc_logits < prev_mc).float()  # [batch,]
        # 假设 penalty_logits 的形状是 [b, n, 2] (每个token有2个类别的logits)
        # 假设 penalty 的形状是 [b, n] (每个token的类别标签，0或1，无效位置为-100)
        
        # F.binary_cross_entropy_with_logits 不支持 ignore_index. 
        # 如果你的 logits 是 [b, n, 2] 并且 targets 是 [b, n] (包含 -100), 
        # 你应该使用 F.cross_entropy，它支持 ignore_index.
        
        # 检查形状是否符合预期
        if penalty_logits.dim() == 3 and penalty_logits.shape[-1] == 2 and penalty.dim() == 2:
            # Reshape logits: [b, n, 2] -> [b*n, 2]
            # Reshape penalty: [b, n] -> [b*n]
            num_classes = penalty_logits.shape[-1]
            penalty_loss = F.cross_entropy(
                penalty_logits.view(-1, num_classes), 
                penalty.view(-1), 
                ignore_index=-100,  # F.cross_entropy 会自动忽略 target 为 -100 的位置
                reduction='mean'  # 计算有效位置的平均损失
            )
        elif penalty_logits.shape == penalty.shape:
            # 如果形状是 [b, n] vs [b, n] 或 [b,] vs [b,] (适用于二元分类 logits)
            # 并且 penalty 包含 -100，你需要手动处理
            valid_mask = (penalty != -100)
            if valid_mask.sum() > 0:
                # 只选择有效位置的 logits 和 targets 进行计算
                valid_logits = penalty_logits[valid_mask]
                valid_penalty = penalty[valid_mask]
                # 确保 targets 是 float 类型
                penalty_loss = F.binary_cross_entropy_with_logits(
                    valid_logits, valid_penalty.float(), reduction='mean'
                )
            else:
                # 如果整个批次都没有有效标签，损失为0
                penalty_loss = torch.tensor(0.0, device=penalty_logits.device, dtype=penalty_logits.dtype)
        else:
            # 如果形状不匹配或不符合上述情况，抛出错误或使用原始逻辑（如果适用）
            raise ValueError(f"Incompatible shapes for penalty_logits ({penalty_logits.shape}) and penalty ({penalty.shape}) for loss calculation with ignore_index.")

        # 原始代码（假设 penalty_logits 和 penalty 形状兼容且 penalty 不含 -100）
        # penalty_loss = F.binary_cross_entropy_with_logits(
        #     penalty_logits, penalty.float() # 确保 target 是 float
        # )

        
        # 5. 组合损失
        total_loss = mc_loss + self.alpha * penalty_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        if isinstance(logits, tuple):
            logits = logits[0] - self.lambda_penalty * logits[1]
        return loss, logits, labels

# class QwenForValueFunction(Qwen2ForSequenceClassification):
#     def __init__(self, config, reduction='mean', cut_ratio=0.5):
#         super().__init__(config)
#         self.reduction = reduction
#         self.cut_ratio=cut_ratio
#         if self.cut_ratio < 1:
#             self.reduction = 'none'
#         # self.activate = torch.nn.functional.
    
#     @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size = input_ids.shape[0]
#         else:
#             batch_size = inputs_embeds.shape[0]

#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#         if self.config.pad_token_id is None:
#             sequence_lengths = -1
#         else:
#             if input_ids is not None:
#                 # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
#                 sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
#                 sequence_lengths = sequence_lengths % input_ids.shape[-1]
#                 sequence_lengths = sequence_lengths.to(logits.device)
#             else:
#                 sequence_lengths = -1

#         pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

#         loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss(reduction=self.reduction)
#                 if self.num_labels == 1:
#                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(pooled_logits, labels)
#                 if self.reduction == 'none':
#                     k = int(self.cut_ratio * batch_size)
#                     smallest_k = torch.topk(loss, k, largest=False).values 
#                     loss = smallest_k.mean()
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(pooled_logits, labels)
#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )



# class LlamaForValueFunction(LlamaForSequenceClassification):
#     def __init__(self, config, reduction='mean', cut_ratio=0.5):
#         super().__init__(config)
#         self.reduction = reduction
#         self.cut_ratio=cut_ratio
#         if self.cut_ratio < 1:
#             self.reduction = 'none'
#         # self.activate = torch.nn.functional.
    
#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)
#         logits = torch.nn.functional.sigmoid(logits)
        
#         if input_ids is not None:
#             batch_size = input_ids.shape[0]
#         else:
#             batch_size = inputs_embeds.shape[0]

#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#         if self.config.pad_token_id is None:
#             sequence_lengths = -1
#         else:
#             if input_ids is not None:
#                 # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
#                 sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
#                 sequence_lengths = sequence_lengths % input_ids.shape[-1]
#                 sequence_lengths = sequence_lengths.to(logits.device)
#             else:
#                 sequence_lengths = -1

#         pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

#         loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss(reduction=self.reduction)
#                 if self.num_labels == 1:
#                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(pooled_logits, labels)
#                 if self.reduction == 'none':
#                     k = int(self.cut_ratio * batch_size)
#                     smallest_k = torch.topk(loss, k, largest=False).values 
#                     loss = smallest_k.mean()
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(pooled_logits, labels)
#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )
        


# def obtain_value_cls(config):
#     architectures = config.architectures[0]
#     if "llama" in architectures.lower():
#         return LlamaForValueFunction
#     elif "qwen" in architectures.lower():
#         return QwenForValueFunction
        
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig
import torch 
from peft import PeftModel

class ValueModel:
    def __init__(self, model_base, model_path, model_type, device_map='auto', **kwargs):
        kwargs = {"device_map": device_map}
        kwargs['torch_dtype'] = torch.bfloat16
        if model_base is None or model_base == 'None':
            if isinstance(model_path, list):
                model_path = model_path[0]
            load_from = model_path 
            lora_path = []
        else:
            load_from = model_base
            lora_path = [model_path] if isinstance(model_path, str) else model_path
            
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path=load_from)
        if config.pad_token_id is None:
            # config.pad_token = config.eos_token
            if "3.1-8b" in load_from.lower() or "3.1-8b" in load_from.lower():
                config.pad_token = "<|finetune_right_pad_id|>"
                config.pad_token_id = 128004
            else:
                config.pad_token_id = config.eos_token_id 
        if model_type == 'prm':
            model = AutoModelForTokenClassification.from_pretrained(load_from, num_labels=2, pad_token_id=config.pad_token_id, **kwargs)
            self.forward_func = self.forward_token
        elif model_type == 'backprm':
            model = ProcessRewardModel.from_pretrained(load_from, num_labels=2, pad_token_id=config.pad_token_id, **kwargs)
            self.forward_func = self.forward_backprm
        elif model_type == 'orm' or model_type == 'prm-bi':
            model = AutoModelForSequenceClassification.from_pretrained(load_from, num_labels=1, pad_token_id=config.pad_token_id, **kwargs)
            self.forward_func = self.forward_sequence
        else:
            raise ValueError(f"Unknown model type {model_type}")
        for path in lora_path:
            model = PeftModel.from_pretrained(model, path)
            model = model.merge_and_unload()
            print(f"Load Lora weights from {path}")
        self.model = model.to(torch.float16)
        self.model_type = model_type
        
    @property
    def device(self):
        return self.model.device
    
    @property
    def dtype(self):
        return self.model.dtype
    
    def forward_sequence(self, input_ids, attention_mask=None, **kwargs):
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # print(outputs[0])
        score = torch.sigmoid(outputs[0]) # [0] means logits
        return score
    
    def forward_backprm(self, input_ids, attention_mask=None, return_all=False):
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if return_all:
            probs = torch.softmax(outputs['logits'], dim=-1)
            penalty_probs = torch.softmax(outputs['penalty_logits'], dim=-1)
        else:
            
            probs = torch.softmax(outputs['logits'][:, -2], dim=-1)
            penalty_probs = torch.softmax(outputs['penalty_logits'][:, -2], dim=-1)
        score = probs[..., 1] # [B, N] or [B, ]
        penalty_score = penalty_probs[..., 1]
        final_score = score - self.model.lambda_penalty * penalty_score 
        return final_score
    
    def forward_token(self, input_ids, attention_mask=None, return_all=False):
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if return_all:
            probs = torch.softmax(outputs[0], dim=-1)
        else:
            
            probs = torch.softmax(outputs[0][:, -1], dim=-1)
        score = probs[..., 1] # [B, N] or [B, ]
        return score
    
    def __call__(self, input_ids, attention_mask=None, **kwargs):
        return self.forward_func(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        
        