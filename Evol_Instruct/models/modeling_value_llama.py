# import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig
import torch 
from peft import PeftModel

class ValueModel:
    def __init__(self, model_base, model_path, model_type, device_map='auto'):
        kwargs = {"device_map": device_map}
        kwargs['torch_dtype'] = torch.float16
        if model_base is None or model_base == 'None':
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
        elif model_type == 'orm':
            model = AutoModelForSequenceClassification.from_pretrained(load_from, num_labels=1, pad_token_id=config.pad_token_id, **kwargs)
            self.forward_func = self.forward_sequence
        else:
            raise ValueError(f"Unknown model type {model_type}")
        for path in lora_path:
            model = PeftModel.from_pretrained(model, path)
            model = model.merge_and_unload()
            print(f"Load Lora weights from {path}")
        self.model = model.to(torch.float16)
        
    @property
    def device(self):
        return self.model.device
    
    @property
    def dtype(self):
        return self.model.dtype
    
    def forward_sequence(self, input_ids, attention_mask=None, **kwargs):
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        score = torch.sigmoid(outputs[0])
        return score
    
    def forward_token(self, input_ids, attention_mask=None, return_all=False):
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if return_all:
            probs = torch.softmax(outputs[0], dim=-1)
        else:
            
            probs = torch.softmax(outputs[0][:, -2], dim=-1)
        score = probs[..., 1] # [B, N] or [B, ]
        return score
    
    def __call__(self, input_ids, attention_mask=None, **kwargs):
        return self.forward_func(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        
        