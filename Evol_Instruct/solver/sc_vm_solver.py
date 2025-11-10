from copy import deepcopy

from Evol_Instruct.MCTS.utils import extract_template

from Evol_Instruct.models.vllm_support import VLLMServer, chat_prompt

from Evol_Instruct.solver.base_solver import Solver
from Evol_Instruct.utils.utils import AlpacaTaskItem, compute_weighted_values
from collections import defaultdict 
from itertools import chain
import numpy as np
import torch

def find_answer_end(s):
    # 寻找 "The answer is" 或 "the answer is"的位置
    pos1 = s.find("The answer is")
    pos2 = s.find("the answer is") 
    
    # 如果都没找到,返回-1
    if pos1 == -1 and pos2 == -1:
        return -1
        
    # 找出实际的起始位置
    if pos1 == -1:
        start_pos = pos2
    elif pos2 == -1:
        start_pos = pos1
    else:
        start_pos = pos1
    
    # 从起始位置往后找.\n\n
    end_pos = s.find(".\n\n", start_pos)
    if end_pos == -1:
        return -1
    return end_pos + 1



#     return step_value
def obtain_prm_value_for_single_pair(tokenizer, value_model, inputs, outputs, server):
    if value_model.model_type == 'prm-bi':
        response = outputs
        completions = [f"Step" + completion if not completion.startswith("Step") else completion for k, completion in enumerate(outputs.split("\n\nStep"))]
        each_sub_response = []
        for i, completion in enumerate(completions):
            each_sub_response.append("\n\n".join(completions[:i+1]) + "\n\n")
        # each_sub_response[-1] += "\n\n"
        input_texts = [] 
        step_value = []
        for i in each_sub_response:
            messages = [
                {"role": "user", "content": inputs},
                {"role": "assistant", "content": i}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False)
            # print(input_text)
            input_texts.append(input_text)
        micro_batch_size = 4
        for i in range(0, len(input_texts), micro_batch_size):
            input_text = input_texts[i: i + micro_batch_size]
            
            input_ids = tokenizer(input_text, 
                              return_tensors='pt',
                              add_special_tokens=True,
                              padding_side='left',
                              padding=True)['input_ids']
            value = value_model(input_ids=input_ids.to(value_model.device))
            step_value.extend(value.squeeze(-1).cpu().numpy().tolist())
        # print(step_value)
        return step_value    

    response = outputs

    
    messages = [
        {"role": "user", "content": inputs},
        {"role": "assistant", "content": response}
    ]
    
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    completions = ["Step" + completion if not completion.startswith("Step") else completion for completion in response.split("\n\nStep")]



    completion_ids = [
        tokenizer(completion + "\n\n", add_special_tokens=False)['input_ids'] for completion in completions
    ]
    response_id = list(chain(*completion_ids))
    pre_response_id = tokenizer(prompt_text, add_special_tokens=False)['input_ids']

    input_ids = pre_response_id + response_id
        
    value = value_model(input_ids=torch.tensor(input_ids).unsqueeze(0).to(value_model.device), return_all=True)  # [1, N]
    
    completion_index = []
    for i, completion in enumerate(completion_ids):
        if i == 0:
            completion_index.append(len(completion) + len(pre_response_id) - 1)
        else:
            completion_index.append(completion_index[-1] + len(completion))
    
    step_value = value[0, completion_index].cpu().numpy().tolist()
    return step_value
    

def obtain_prm_value_for_batch_pair(tokenizer, value_model, inputs, outputs, server):
    bs = len(inputs)
    input_ids_list = []
    completion_index_list = []
    for i in range(bs):
        cur_inputs = inputs[i]
        cur_response = outputs[i]
        messages = [
            {"role": "user", "content": cur_inputs},
            {"role": "assistant", "content": cur_response}
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        completions = ["Step" + completion if not completion.startswith("Step") else completion for completion in cur_response.split("\n\nStep")]
        # input_text = prompt_text + response + "\n\n"



        completion_ids = [
            tokenizer(completion + "\n\n", add_special_tokens=False)['input_ids'] for completion in completions
        ]
        response_id = list(chain(*completion_ids))
        pre_response_id = tokenizer(prompt_text, add_special_tokens=False)['input_ids']

        input_ids = pre_response_id + response_id
            
        
        completion_index = []
        for i, completion in enumerate(completion_ids):
            if i == 0:
                completion_index.append(len(completion) + len(pre_response_id) - 1)
            else:
                completion_index.append(completion_index[-1] + len(completion))
        input_ids_list.append(input_ids)
        completion_index_list.append(completion_index)
    
    # input_ids_list should be right pad to the same length, and obtain corresponding attention mask matrix
    max_len = max([len(x) for x in input_ids_list])
    attention_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in input_ids_list]
    pad_token_id = tokenizer.pad_token_id
    input_ids_list = [x + [pad_token_id] * (max_len - len(x)) for x in input_ids_list]
    
    input_ids = torch.tensor(input_ids_list).to(value_model.device)
    attention_mask = torch.tensor(attention_mask).to(value_model.device)
    value = value_model(input_ids=input_ids, attention_mask=attention_mask, return_all=True)  # [B, N]
    
    step_value_list = []
    for i in range(bs):
        completion_index = completion_index_list[i]
        step_value = value[0, completion_index].cpu().numpy().tolist()
        step_value_list.append(step_value)
    return step_value_list


class SCVMSolver(Solver):
    def __init__(self, *args, value_model, infer_rule, **kwargs):
        """
        server: VLLMServer, a server to handle model inference
        save_file: TextIOWrapper, a file handler
        choices_word: list, a list of choice words
        cot_prompt: str, the prompt to infer answer
        
        """
        super().__init__(*args, **kwargs)
        self.value_model = value_model
        self.infer_rule = infer_rule 
        
    def obtain_orm_value(self, inputs, outputs):
        bs = len(inputs)
        conversations = [[
            {"role": "user", "content": x},
            {"role": "assistant", "content": y}
        ] for x, y in zip(inputs, outputs)]
        
        texts = self.server.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        tokens = [self.server.tokenizer(x, return_tensors='pt',)['input_ids'] for x in texts]

        values = []
        for i in range(0, len(tokens)):
            cur_id = tokens[i].to(self.value_model.device)
            value = self.value_model(input_ids=cur_id,)
            values.append(value)
        value_model_scores = torch.cat(values, dim=0)
        # values = torch.cat(values, dim=0)
        assert value_model_scores.shape == (bs, 1)
            # values = self.value_model(**tokens)
        # value_model_scores = torch.sigmoid(values[:, 0]) # [B*N, ]
        return value_model_scores[:, 0]
    
    def compute_score_loc(self, inputs, outputs):
        messages = [
            {"role": "user","content": inputs},
            {"role": "assistant", "content": outputs}
        ]
        completions = [f"Step" + completion if not completion.startswith("Step") else completion for k, completion in enumerate(outputs.split("\n\nStep"))]
        
        prompt_text = self.server.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = self.server.tokenizer(prompt_text)['input_ids']
        completion_ids = [
            self.server.tokenizer(completion, add_special_tokens=False)['input_ids'] for completion in completions
        ]
        # completion_labels = [[-100] * (len(completion) - 1) + [1] for completion in completion_ids]
        start_idx = len(prompt_ids)
        score_loc = []
        for i, completion in enumerate(completion_ids):
            score_loc.append(start_idx + sum([len(completion_ids[j]) for j in range(0, i)]) - 1 + len(completion))
            

        return score_loc
        
    
    def obtain_prm_values(self, inputs, outputs):
        bs = len(inputs)
        values = []

        for i in range(bs):
            cur_input = inputs[i]
            cur_output = outputs[i]
            loc_value = obtain_prm_value_for_single_pair(self.server.tokenizer, self.value_model,
                                                         cur_input, cur_output, self.server)
            values.append(loc_value)
        return values
    
        
    
    def orm_select(self, inputs, outputs, answer_outputs, batch, n):
        values = self.obtain_orm_value(inputs, outputs)
        outputs = [[answer_outputs[i * n + j] for j in range(n)] for i in range(batch)] 
        values = [[values[i * n + j] for j in range(n)] for i in range(batch)]
        only_answer_outputs = [[extract_template(x, 'answer') for x in y] for y in outputs]
        vote_output = []
        for i in range(batch):
            weighted_values = defaultdict(float)
            count_values = defaultdict(int)
            max_answer, weighted_values = compute_weighted_values(only_answer_outputs[i], values[i], self.infer_rule)
        
            max_weighted_value = weighted_values[max_answer]
            tie_answers = [ans for ans, val in weighted_values.items() if val == max_weighted_value]
            if len(tie_answers) > 1:
                # max_answer = tie_answers[0]
                best_answer = max(((outputs[i][j], values[i][j]) for j in range(n) if extract_template(outputs[i][j], 'answer') == max_answer), key=lambda n: n[1])
                best_answer = best_answer[0]
            else:
                best_answer = None
                best_value = float("-inf")
                for j in range(n):
                    if extract_template(outputs[i][j], 'answer') in tie_answers and values[i][j] > best_value:
                        best_answer = outputs[i][j]
                        best_value = values[i][j]
            vote_output.append(best_answer)
        return vote_output, only_answer_outputs, values
    
    def prm_select(self, inputs, outputs, answer_outputs, batch, n):
        values = self.obtain_prm_values(inputs, outputs)
        outputs = [[answer_outputs[i * n + j] for j in range(n)] for i in range(batch)] 
        values = [[values[i * n + j] for j in range(n)] for i in range(batch)]
        original_values = deepcopy(values)
        only_answer_outputs = [[extract_template(x, 'answer').strip().strip("*").strip(":") if extract_template(x, 'answer') is not None else "" for x in y] for y in outputs]
        
        vote_output = []
        for i in range(batch):
            max_answer, weighted_values = compute_weighted_values(only_answer_outputs[i], values[i], self.infer_rule)
            value_op = self.infer_rule.split("-")[1]
            if value_op == 'prod':
                values[i] = [np.prod(x[1:]) for x in values[i]]
            elif value_op == 'mean':
                values[i] = [np.mean(x[1:]) for x in values[i]]
            elif value_op == 'vote':
                values[i] = [x[-1] for x in values[i]]
            elif value_op == 'min':
                values[i] = [np.min(x[1:]) if len(x) > 1 else 0 for x in values[i]]
            elif value_op == 'max':
                values[i] = [x[-1] for x in values[i]]
            else:
                raise ValueError(f"PRM value operators not support {value_op}")
            
            max_weighted_value = weighted_values[max_answer]
            tie_answers = [ans for ans, val in weighted_values.items() if val == max_weighted_value]
            if len(tie_answers) > 1:
                # max_answer = tie_answers[0]
                best_answer = max(((outputs[i][j], values[i][j]) for j in range(n) if extract_template(outputs[i][j], 'answer') == max_answer), key=lambda n: n[1])
                best_answer = best_answer[0]
            else:
                best_answer = None
                best_value = float("-inf")
                for j in range(n):
                    if extract_template(outputs[i][j], 'answer') in tie_answers and values[i][j] > best_value:
                        best_answer = outputs[i][j]
                        best_value = values[i][j]
            vote_output.append(best_answer)
        return vote_output, only_answer_outputs, original_values
            
        
    def generate_response(self, items: list[AlpacaTaskItem], **kwargs):
        """
        temperature: float
        n: int, the sample number
        max_tokens: int, the maximum tokens to generate
        top_p: float, the top p value
        value_model_type: str, the value model type to use (prm or orm)
        """
        temperature = kwargs.pop("temperature", 1.0)
        n = kwargs.pop("n", 16)
        max_tokens = kwargs.pop("max_tokens", 1024)
        top_p = kwargs.pop("top_p", 0.95)
        value_model_type = kwargs.pop("value_model_type", "orm")
        
        prompts = [chat_prompt([item.prompt], self.server.tokenizer,
                               system='You are a helpful medical assistant.')[0] for item in items]
        
        # use mini-batch (16) to inference and re-construct
        mini_batch = 16
        outputs = []
        for i in range(0, len(prompts), mini_batch):
            outputs += self.server(
                prompts[i: i + mini_batch],
                wrap_chat=False,
                system='You are a helpful medical assistant.',
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                top_p=top_p
            )
        
        outputs = [y for x in outputs for y in x]
        inputs = [item.prompt for item in items for _ in range(n)] # [B x N]
        
        
                
        
        answer_outputs = outputs
        
        batch = len(items)
        
        if value_model_type == 'orm':
            vote_output, only_answer_outputs, values = self.orm_select(inputs, outputs, answer_outputs, batch, n)
            
        elif value_model_type == 'prm' or value_model_type == 'prm-bi':
            vote_output, only_answer_outputs, values = self.prm_select(inputs, outputs, answer_outputs, batch, n)
                # answer_count = Counter(only_answer_outputs[i])
        else:
            raise NotImplementedError
        
        
        for i, item in enumerate(items):
            item.text = vote_output[i]   
            # itm.all_answer = [(a, v.item() if not isinstance(v, list) else v) for a, v in zip(only_answer_outputs[i], values[i])]
            item.all_answer = [(a, v.item() if not isinstance(v, list) else v) for a, v in zip(only_answer_outputs[i], values[i])]
            item.all_traj = answer_outputs[i * n: (i + 1) * n]

        
        return items
    
    
    def rescore(self, items: list[dict[str]], **kwargs):
        if "all_traj" not in items[0]:
            pass 
        batch = len(items)
        
        value_model_type = kwargs.pop("value_model_type", "orm")
        n = len(items[0]['all_traj'])
        inputs = [item['prompt'] for item in items for _ in range(n)] # [B x N]
        outputs = [item['all_traj'][i] for item in items for i in range(n)]
        answer_outputs = outputs
        if value_model_type == 'orm':
            vote_output, only_answer_outputs, values = self.orm_select(inputs, outputs, answer_outputs, batch, n)
            
        elif value_model_type == 'prm' or value_model_type == 'prm-bi':
            vote_output, only_answer_outputs, values = self.prm_select(inputs, outputs, answer_outputs, batch, n)
                # answer_count = Counter(only_answer_outputs[i])
        else:
            raise NotImplementedError
        
        for i, item in enumerate(items):
            item['text'] = vote_output[i]   

            item['all_answer'] = [(a, v.item() if not isinstance(v, list) else v) for a, v in zip(only_answer_outputs[i], values[i])]

        return items
