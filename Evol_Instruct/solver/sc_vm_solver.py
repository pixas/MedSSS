from copy import deepcopy

from regex import I
from Evol_Instruct.MCTS.utils import extract_template
from Evol_Instruct.evaluation.eval_em import extract_answer_content
from Evol_Instruct.evaluation.generate_utils import set_tokenizer
from Evol_Instruct.models.vllm_support import VLLMServer, chat_prompt, vllm_clean_generate

from Evol_Instruct.solver.base_solver import Solver
from Evol_Instruct.utils.utils import AlpacaTaskItem, extract_answer, compute_weighted_values
from collections import Counter, defaultdict 
from itertools import chain
import numpy as np
import pdb
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



def obtain_prm_value_for_single_pair(tokenizer, value_model, inputs, outputs, server):
    # 
    response = outputs
    completions = [f"Step" + completion if not completion.startswith("Step") else completion for k, completion in enumerate(outputs.split("\n\nStep"))]
    
    messages = [
        {"role": "user", "content": inputs},
        {"role": "assistant", "content": response}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    try:
        response_begin_index = input_text.index(response)
    except:
        print(response)
        response = response.rstrip().rstrip("\n\n")
        # print(server(inputs,
        #     wrap_chat=True,
        #     system='You are a helpful medical assistant.',
        #     temperature=0.7,
        #     n=16,
        #     max_tokens=8192,
        #     top_p=0.95))
        # exit(-1)
        response_begin_index = input_text.index(response)
        

    pre_response_input = input_text[:response_begin_index]
    after_response_input = input_text[response_begin_index + len(response):]
    completion_ids = [
        tokenizer(completion + "\n\n", add_special_tokens=False)['input_ids'] for completion in completions
    ]
    
    response_id = list(chain(*completion_ids))
    pre_response_id = tokenizer(pre_response_input, add_special_tokens=False)['input_ids']
    after_response_id = tokenizer(after_response_input, add_special_tokens=False)['input_ids']

    
    input_ids = pre_response_id + response_id + after_response_id
    
    value = value_model(input_ids=torch.tensor(input_ids).unsqueeze(0).to(value_model.device), return_all=True)  # [1, N]
    
    completion_index = []
    for i, completion in enumerate(completion_ids):
        if i == 0:
            completion_index.append(len(completion) + len(pre_response_id) - 1)
        else:
            completion_index.append(completion_index[-1] + len(completion))
    
    step_value = value[0, completion_index].cpu().numpy().tolist()
    return step_value
    
    

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
            
        # total_completion = self.server.tokenizer.apply_chat_template(messages, tokenize=False)
        # total_input_ids = self.server.tokenizer(total_completion)['input_ids']
        
        # score_loc = [start_idx + len(completion) for completion in completion_ids]
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
    
        # conversations = [[
        #     {"role": "user", "content": x},
        #     {"role": "assistant", "content": y}
        # ] for x, y in zip(inputs, outputs)]
        
        # texts = self.server.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        # tokens = [self.server.tokenizer(x, return_tensors='pt',)['input_ids'] for x in texts]

        # values = []
        # for i in range(0, len(tokens)):
        #     cur_id = tokens[i].to(self.value_model.device)
        #     value = self.value_model(input_ids=cur_id, return_all=True)
        #     cur_output = outputs[i]
        #     cur_input = inputs[i]
        #     score_loc = self.compute_score_loc(cur_input, cur_output)
        #     loc_value = [value[0, i].item() for i in score_loc]
        #     values.append(loc_value)
        # # value_model_scores = torch.cat(values, dim=0)
        # # values = torch.cat(values, dim=0)
        # # assert value_model_scores.shape == (bs, 1)
        #     # values = self.value_model(**tokens)
        # # value_model_scores = torch.sigmoid(values[:, 0]) # [B*N, ]
        # return values
    
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
            # for j in range(n):
            #     weighted_values[only_answer_outputs[i][j]] += values[i][j]
            #     count_values[only_answer_outputs[i][j]] += 1
            # if self.infer_rule == 'vote-mean':
            #     weighted_values = {k: v / count_values[k] for k, v in weighted_values.items()}
            # elif self.infer_rule == 'vote-sum':
            #     weighted_values = weighted_values
            # else:
            #     raise NotImplementedError
            # max_answer = max(weighted_values, key=weighted_values.get)
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
        only_answer_outputs = [[extract_template(x, 'answer') for x in y] for y in outputs]
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
                values[i] = [np.min(x[1:]) for x in values[i]]
            else:
                raise ValueError(f"PRM value operators not support {value_op}")
            # weighted_values = defaultdict(float)
            # count_values = defaultdict(int)
            # # trajectory_values = defaultdict(float)
            
            # if self.infer_rule == 'prm-times-vote-sum':
                

            #     for j in range(n):
            #         values[i][j] = np.prod(values[i][j])
            #         weighted_values[only_answer_outputs[i][j]] += values[i][j]
            #         count_values[only_answer_outputs[i][j]] += 1
                    
            # elif self.infer_rule == 'prm-times-vote-mean':

            #     for j in range(n):
            #         values[i][j] = min(values[i][j])
            #         weighted_values[only_answer_outputs[i][j]] += values[i][j]
            #         count_values[only_answer_outputs[i][j]] += 1
            #     weighted_values = {k: v / count_values[k] for k, v in weighted_values.items()}
            # elif self.infer_rule == 'prm-mean-vote-sum':
            #     for j in range(n):
            #         values[i][j] = np.mean(values[i][j])
            #         weighted_values[only_answer_outputs[i][j]] += values[i][j]
            #         count_values[only_answer_outputs[i][j]] += 1
            # elif self.infer_rule == 'prm-mean-vote-mean':
            #     for j in range(n):
            #         values[i][j] = np.mean(values[i][j])
            #         weighted_values[only_answer_outputs[i][j]] += values[i][j]
            #         count_values[only_answer_outputs[i][j]] += 1
            #     weighted_values = {k: v / count_values[k] for k, v in weighted_values.items()}
            # else:
                
            #     raise NotImplementedError
            # max_answer = max(weighted_values, key=weighted_values.get)
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
        
        outputs = self.server(
            prompts,
            wrap_chat=False,
            system='You are a helpful medical assistant.',
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_p=top_p   
        )
        # [B x N]
        outputs = [y for x in outputs for y in x]
        inputs = [item.prompt for item in items for _ in range(n)] # [B x N]
        
        
        # generated_answer = [extract_answer_content(x) for x in outputs]
        # no_answer_index = [i for i, x in enumerate(generated_answer) if x is None]
        
        # no_answer_outputs = [outputs[i] for i in no_answer_index]
        # no_answer_prompts = [prompts[i // n] for i in no_answer_index]
        # if no_answer_index:
        #     answer_outputs = self.infer_answer(
        #         no_answer_prompts,
        #         no_answer_outputs,
        #         choices_word=self.choices_word,
        #         cot_prompt=self.cot_prompt
        #     )
        #     for i, index in enumerate(no_answer_index):
        #         outputs[index] = answer_outputs[i]
        # for i, output in enumerate(outputs):
        #     # for each sample in output, cut off repetition words
        #     # for i, sample in enumerate(output):
        #         # identify the first place of 'the answer is '
        #     end_pos = find_answer_end(output)
        #     outputs[i] = output[:end_pos]
                
        
        answer_outputs = outputs
        
        batch = len(items)
        
        if value_model_type == 'orm':
            vote_output, only_answer_outputs, values = self.orm_select(inputs, outputs, answer_outputs, batch, n)
            
        elif value_model_type == 'prm':
            vote_output, only_answer_outputs, values = self.prm_select(inputs, outputs, answer_outputs, batch, n)
                # answer_count = Counter(only_answer_outputs[i])
        else:
            raise NotImplementedError
        
        
        for i, item in enumerate(items):
            item.text = vote_output[i]   
            item.all_answer = [(a, v.item() if not isinstance(v, list) else v) for a, v in zip(only_answer_outputs[i], values[i])]
        # batch_answer_content = [[extract_answer_content(x)[0] for x in y] for y in outputs]
        # batch_answer_index = [row.index(Counter(row).most_common(1)[0][0]) for row in batch_answer_content]

        # # print(batch_answer_content)
        # outputs = [y[x] for x, y in zip(batch_answer_index, outputs)]
        # for i, item in enumerate(items):
        #     item.text = outputs[i]
        
        return items

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from Evol_Instruct import client
    from Evol_Instruct.evaluation.generate_utils import CustomDataset
    from Evol_Instruct.models.modeling_value_llama import ValueModel
    data = client.read("/mnt/petrelfs/jiangshuyang.p/datasets/medical_test/medsins_task131_500.json")
    model_base='/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3.1-8B-Instruct-ysl'
    lora_path='/mnt/petrelfs/jiangshuyang.p//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-SFT-ITER1'
    reward_model_path = '/mnt/petrelfs/jiangshuyang.p/checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-VALUE-prm_trainall_r64_softtrain_basepolicy-ITER1'
    # tokenizer = AutoTokenizer.from_pretrained(model_base)
    # print(tokenizer.)
    # set_tokenizer(tokenizer)
    server = VLLMServer(url='http://10.140.1.163:10002', 
                        model=model_base,
                        lora_path=lora_path,
                        offline=True, gpu_memory_usage=0.45, max_model_len=16384)
    tokenizer = server.tokenizer
    print(tokenizer)
    value_model = ValueModel(model_base, [lora_path, reward_model_path], 'prm')
    
    dataset = CustomDataset(data, 1, "", tokenizer=tokenizer, )
    solver = SCVMSolver(server, open('temp.jsonl', 'w'), ['A', 'B', 'C', 'D', 'E'], value_model=value_model, infer_rule='prm-vote-sum', cot_prompt="\nThe answer is ")
    
    while 1:
        idx = int(input())
        output = solver.generate_response(dataset[idx], value_model_type='prm')
        solver.save_response(output)
        print("generate one instance over", flush=True)