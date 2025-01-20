from calendar import c
import numpy as np
import torch
from Evol_Instruct.evaluation.eval_em import extract_answer_content
from Evol_Instruct.models.vllm_support import chat_prompt, vllm_clean_generate
from Evol_Instruct.solver.base_solver import Solver
from Evol_Instruct.utils.utils import AlpacaTaskItem
from Evol_Instruct.evaluation.generate_utils import infer_answer
from collections import defaultdict 
from copy import deepcopy

def dpo_filter(dpo_model, questions: list[str], outputs: list[list[str]], tokenizer, small_batch_size=4):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    # Construct BxK prompts
    convs = []
    for question, output_list in zip(questions, outputs):
        for output in output_list:
            new_messages = deepcopy(messages)
            new_messages.append({"role": "user", "content": question})
            new_messages.append({"role": "assistant", "content": output})
            convs.append(new_messages)
            
    # Tokenize all prompts without padding initially
    prompts = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True)
    
    # Split the inputs into small batches for inference
    likelihoods = []
    device = dpo_model.device
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    for i in range(0, input_ids.size(0), small_batch_size):
        # Slice small batch
        small_input_ids = input_ids[i:i+small_batch_size].to(device)
        small_attention_mask = attention_mask[i:i+small_batch_size].to(device)
        
        with torch.inference_mode():
            outputs = dpo_model(input_ids=small_input_ids, attention_mask=small_attention_mask)
            
            # [B*K, N, V]
            logits = outputs.logits
            
            # Gather logits for the target tokens
            target_logits = torch.gather(logits, 2, small_input_ids.unsqueeze(-1)).squeeze(-1)  # [small_batch_size, N]
            
            # Calculate logsumexp along the vocabulary dimension for each token
            logsumexp_logits = torch.logsumexp(logits, dim=-1)  # [small_batch_size, N]
            
            # Compute the log probability of each target token
            log_probability = target_logits - logsumexp_logits  # [small_batch_size, N]
            
            # Apply attention mask to ignore padding tokens
            masked_log_probability = log_probability * small_attention_mask.float()  # [small_batch_size, N]
            
            # Compute sequence likelihood by summing log probabilities and normalizing
            batch_likelihood = (masked_log_probability.sum(dim=1) / small_attention_mask.sum(dim=1))  # [small_batch_size]
            likelihoods.append(batch_likelihood)
    
    # Concatenate all batch likelihoods
    likelihoods = torch.cat(likelihoods)
    
    # Reshape likelihoods to [B, K] and find the index of the max likelihood for each question
    max_likelihood_index = likelihoods.reshape(len(questions), -1).argmax(dim=-1)
    
    return max_likelihood_index, likelihoods

class DPOJudgeSolver(Solver):
    def __init__(self, *args, dpo_model, dpo_select_method='vote', **kwargs):
        super().__init__(*args, **kwargs)
        self.dpo_model = dpo_model
        self.dpo_select_method = dpo_select_method
        # print(args, kwargs)
        
    def generate_response(self, items: list[AlpacaTaskItem], **kwargs):
        temperature = kwargs.pop("temperature", 1.0)
        n = kwargs.pop("n", 16)
        max_tokens = kwargs.pop("max_tokens", 1024)
        top_p = kwargs.pop("top_p", 0.95)
        
        prompts = [chat_prompt([item.prompt], self.server.tokenizer,
                               system='You are a helpful medical assistant.')[0] for item in items]
        outputs = self.server(
            prompts,
            wrap_chat=False,
            system='You are a helpful medical assistant.',
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_p=top_p   
        )
        
        if self.dpo_select_method == 'max':
            max_likelihood_index, _ = dpo_filter(self.dpo_model,
                                                 prompts,
                                                 outputs,
                                                 self.server.tokenizer)
            outputs = [output[max_likelihood_index[i]] for i, output in enumerate(outputs)]
        else:
            _, likelihood = dpo_filter(self.dpo_model, prompts, outputs, self.server.tokenizer)
            outputs = [y for x in outputs for y in x]
        answer_outputs = self.infer_answer(
            prompts,
            outputs,
            choices_word=self.choices_word,
            cot_prompt=self.cot_prompt
        )
        # answer_outputs = infer_answer(
        #     self.tokenizer,
        #     self.model,
        #     prompts,
        #     outputs,
        #     self.lora_request,
        #     choices_word=self.choices_word,
        #     cot_prompt=self.cot_prompt
        # )
        batch = len(items)
        if self.dpo_select_method == 'vote':
            outputs = [[answer_outputs[i * n + j] for j in range(n)] for i in range(batch)]
            likelihood = likelihood.reshape(len(prompts), -1)  # [B, K]
            probability = torch.exp(likelihood)
            batch_answer_content = [[extract_answer_content(x)[0] for x in y] for y in outputs]
            results = []
            for answers, likeli in zip(batch_answer_content, probability):
                answer_likelihoods = defaultdict(list)
                for answer, lik in zip(answers, likeli):
                    answer_likelihoods[answer].append(lik.detach().cpu().numpy())
                avg_likelihoods = {answer: np.sum(vals) for answer, vals in answer_likelihoods.items()}
                
                # Step 2: Find the answer with the maximum average likelihood and get any one of its indexes
                max_answer = max(avg_likelihoods, key=avg_likelihoods.get)
                max_index = next(i for i, ans in enumerate(answers) if ans == max_answer)
                
                # Store the results for this batch
                results.append({
                    'average_likelihoods': avg_likelihoods,
                    'max_answer_index': max_index
                })
            outputs = [y[x['max_answer_index']] for x, y in zip(results, outputs)]
        else:
            outputs = answer_outputs 
        for i, item in enumerate(items):
            item.text = outputs[i]
        return items