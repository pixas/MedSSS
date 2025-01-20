from vllm import LLM, SamplingParams
from transformers import AutoConfig
from peft import LoraConfig
from vllm.lora.request import LoRARequest
from openai import OpenAI
import pdb
import torch 
from transformers import LogitsProcessor

from Evol_Instruct.utils.utils import proxy_manager, LogitBiasProcess


def get_vllm_model(model_path, **kwargs):
    """ obtain the vllm model with the specified parameters
    Args
        * model_path: str
        * quantization: bool
        * lora_path: str, the lora path to use in the model
        * gpu_memory_usage: float, the gpu memory proportion to use
        * max_model_len: int, the maximum model length to use
    
    """
    model_config = AutoConfig.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    quantization = kwargs.pop("quantization", None)
    lora_path = kwargs.pop("lora_path", None)
    gpu_memory_usage = kwargs.pop("gpu_memory_usage", 0.9)
    max_model_len = kwargs.pop("max_model_len", 32000)
    if lora_path is not None:
        lora_config = LoraConfig.from_pretrained(lora_path)
        r = lora_config.r 
        model = LLM(model=model_path, max_model_len=min(max_model_len, model_config.max_position_embeddings), gpu_memory_utilization=gpu_memory_usage, quantization=quantization,
                    enable_lora=True,
                    max_lora_rank=r, trust_remote_code=True)
        
        lora_adapter = LoRARequest("adapter", 1, lora_path)
    else:
        lora_adapter = None
        model = LLM(model=model_path, max_model_len=min(max_model_len, model_config.max_position_embeddings), gpu_memory_utilization=gpu_memory_usage, quantization=quantization, trust_remote_code=True)

    return model, lora_adapter

def chat_prompt(prompts, tokenizer, system=None) -> list[str]:
    new_prompt = []
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        template = [
            {
                "role": "system",
                "content": "You are a helpful medical assistant to help patients. Make sure to provide precise, accurate and snappy responses." if system is None else system
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        text = tokenizer.apply_chat_template(
                                template,
                                tokenize=False,
                                add_generation_prompt=True
                            )
        new_prompt.append(text)
    return new_prompt


def vllm_generate(model, tokenizer, prompts, wrap_chat=True, **sampling_params):
    if isinstance(prompts, str):
        prompts = [prompts]
    if wrap_chat:
        prompts = chat_prompt(prompts, tokenizer)
    terminators = [
        tokenizer.eos_token,
        "<|eot_id|>"
    ]
    temperature = sampling_params.pop("temperature", 1)
    top_p = sampling_params.pop("top_p", 0.95)
    max_tokens = sampling_params.pop("max_tokens", 4096)
    n = sampling_params.pop("n", 4)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, stop=terminators, max_tokens=max_tokens, n=n, **sampling_params)
    outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    # regenerated_prompts = []
    generated_texts = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        all_outputs = output.outputs
        for output in all_outputs:
            if "to create" in output.text.lower() or "version" in output.text.lower() or "here's" in output.text.lower():
                # print(f"prompt: {prompt}, output: {output.text}")
                continue
            generated_text = output.text
            # print(f"prompt: {prompt}, generated_text: {generated_text}")
            generated_texts.append(generated_text)
            break
        # generated_text = output.outputs[0].text 
        # # repeat = 0
        # if "to create" in generated_text.lower() or "here is " in generated_text.lower() or "here's" in generated_text.lower():
        #     new_sampled_params = sampling_params.clone()
        #     new_sampled_params.n = 3
        #     new_sampled_params.best_of = new_sampled_params.n
        #     outputs = model.generate([prompt], new_sampled_params, use_tqdm=False)[0].outputs
        #     for output in outputs:
        #         if "to create" in output.text.lower() or "here is " in output.text.lower() or "here's" in output.text.lower():
        #             continue
        #         generated_text = output.text
        #         break
            # repeat += 1
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # generated_texts.append(generated_text) 
    return generated_texts 


def vllm_clean_generate(model, prompts, system=None, wrap_chat=True, lora_request=None, **sampling_params):
    if isinstance(prompts, str):
        prompts = [prompts]
    tokenizer = model.get_tokenizer()
    if wrap_chat:
        prompts = chat_prompt(prompts, tokenizer, system=system)


    temperature = sampling_params.pop("temperature", 1)
    top_p = sampling_params.pop("top_p", 0.95)
    max_tokens = sampling_params.pop("max_tokens", 4096)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, **sampling_params)
    output = model.generate(prompts, sampling_params, use_tqdm=False, lora_request=lora_request)
    generated_texts = [[x.text for x in all_outputs.outputs] for all_outputs in output]
    return generated_texts


def vllm_clean_multiround_generate(model, conversations, system=None, **sampling_params):
    tokenizer = model.get_tokenizer()
    new_conversations = []
    for conversation in conversations:
        new_conversation = []
        for turn in conversation:
            if turn["role"] == "system":
                new_conversation.append({
                    "role": "system",
                    "content": "You are a helpful assistance" if system is None else system
                })
            elif turn['role'] == 'user':
                new_conversation.append({
                    "role": "user",
                    "content": turn["content"]
                })
            else:
                new_conversation.append({
                    "role": "assistant",
                    "content": turn['content']
                })
        text = tokenizer.apply_chat_template(
                                new_conversation,
                                tokenize=False,
                                add_generation_prompt=True
                            )
        new_conversations.append(text)
    temperature = sampling_params.pop("temperature", 1)
    top_p = sampling_params.pop("top_p", 0.95)
    max_tokens = sampling_params.pop("max_tokens", 4096)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, **sampling_params)
    output = model.generate(new_conversations, sampling_params, use_tqdm=False)
    generated_texts = [[x.text for x in all_outputs.outputs] for all_outputs in output]

    return generated_texts

class VLLMServer:
    def __init__(self, url, model, tokenizer=None, offline=False, **kwargs):
        if offline:
            self.model, self.lora_request = get_vllm_model(model, **kwargs)
        else:
            self.client = OpenAI(
                base_url=f"{url}/v1"
            )
            self.model = model
            self.lora_request = None
        self.tokenizer = tokenizer if tokenizer is not None else self.model.get_tokenizer()
        self.offline = offline
    
    def __call__(self, prompt, system=None, wrap_chat=True, **sampling_params):
        if self.offline:
            return vllm_clean_generate(self.model, prompt, system=system, wrap_chat=wrap_chat, lora_request=self.lora_request, **sampling_params)
        
        if isinstance(prompt, str):
            prompt = [prompt]
        if wrap_chat:
            prompt = chat_prompt(prompt, self.tokenizer, system=system)
        temperature = sampling_params.pop("temperature", 1)
        top_p = sampling_params.pop("top_p", 0.95)
        max_tokens = sampling_params.pop("max_tokens", 4096)
        logits_processors = sampling_params.pop("logits_processors", None)
        if logits_processors is not None:
            # pdb.set_trace()
            logit_bias_processor = [x for x in logits_processors if isinstance(x, LogitBiasProcess)][0]
            token_ids = logit_bias_processor.activate_token_list
            # tokens = [self.tokenizer.decode(x) for x in token_ids]
            logit_bias = {token: logit_bias_processor.activate_scale for token in token_ids}
        else:
            logit_bias = None
        with proxy_manager():
            responses = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logit_bias=logit_bias,
                **sampling_params
            )
        n = sampling_params.pop("n", 1)
        generated_texts = [[responses.choices[i * n + j].text for j in range(n)] for i in range(len(prompt))]
        # pdb.set_trace()
        return generated_texts

if __name__ == "__main__":
    from transformers import AutoTokenizer
    model_path = "/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    server = VLLMServer("http://10.140.0.170:10002", model_path, tokenizer)
    output = server(["""You are a professional medical expert majored at reasoning in hard medical-related problems.\n\n<problem>\nA mother brings her 3-week-old infant to the pediatrician\'s office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child\'s abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?\nA. Abnormal migration of ventral pancreatic bud\nB. Complete failure of proximal duodenum to recanalize\nC. Abnormal hypertrophy of the pylorus\nD. Failure of lateral body folds to move ventrally and fuse in the midline\n</problem>\n\n<reasoning_steps>\nStep0: Let\'s break down this problem step by step.\n</reasoning_steps>\n\nGiven all previous reasoning steps, decide on the most suitable action to take next for solving the given problem:\n\n1. Reason: if the task requires further one-step reasoning or no previous reasoning steps available. Must choose this if no previous reasoning steps are available.\n2. Reflect: if the previous reasoning steps contain ambiguity or mistakes. Only choose "Reflect" if there are prior reasoning steps.\n3. Finish: if all reasoning steps are complete enough and a final answer can be provided. Only choose "Finish" if there are two or more reasoning steps.\nOutput your choice using the format: "The action is {{action}}." {{action}} can only takes \'Reason\', \'Reflect\', or \'Finish\'. Do not output other information."""], n=3, stop=["</steps>"])
    