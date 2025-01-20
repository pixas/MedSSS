from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List


ultrarm_template = """Human: {instruction}

Assistant: {completion}"""

def ultra_rm_apply_chat_template(messages):
    return ultrarm_template.format(instruction=messages[0], completion=messages[1])

def set_ultra_rm_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}\n\n\n{% else %}Assistant: {{ message['content'] }}{% endif %}{% endfor %}"
    # tokenizer.apply_chat_template = ultra_rm_apply_chat_template
    return tokenizer

def set_pairrm_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"
    return tokenizer

pairrm_prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"