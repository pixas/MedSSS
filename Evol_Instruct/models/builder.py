#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os

from Evol_Instruct import logger
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch


from Evol_Instruct import client
from peft import PeftModel
from Evol_Instruct.models.modeling_value_llama import ValueModel
from Evol_Instruct.models.vllm_support import VLLMServer




def load_pretrained_model(model_path, model_base, model_name, args, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):


    alpha = getattr(args, "alpha", None)


    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    load_from = model_base if (model_base is not None and model_base != 'None') else model_path
    config = AutoConfig.from_pretrained(load_from)
    
    max_model_len = config.max_position_embeddings
    if max_model_len > 8192:
        max_model_len = 16384
   
    if model_base is not None and model_base != "None":

        server = VLLMServer(None, model_base, None, offline=True, 
                            lora_path=model_path, gpu_memory_usage=args.gpu_memory_usage, max_model_len=max_model_len)
    else:
        server = VLLMServer(None, model_path, None, offline=True, gpu_memory_usage=args.gpu_memory_usage, max_model_len=max_model_len)



        
    return server

def load_dpo_model(model_path, model_base=None, load_from='base', device_map='auto'):
    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.float16
    
    if model_base is None:
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # lora mode
        if load_from == 'base':
            model_base = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
        else:
            # model_base = model
            model_base = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            model = PeftModel.from_pretrained(model_base, load_from)
            model_base = model.merge_and_unload()
            
        model = PeftModel.from_pretrained(model_base, model_path)
        model = model.merge_and_unload()
    
    model = model.to(torch.float16)
    logger.info("Load DPO model successfully")
    return model

def load_value_model(model_path, model_base, model_type, device_map='auto'):
    model = ValueModel(model_base, model_path, model_type, device_map)
    
    return model
    