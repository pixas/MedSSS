import torch 
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from copy import deepcopy
import math
from Evol_Instruct.models.vllm_support import vllm_clean_generate
from Evol_Instruct.utils.utils import AlpacaTaskItem, LogitBiasProcess

task_specific_prompt_mapping = {
    'apps': "\n\nPlease use python language to answer this problem. You should process stdin and stdout with input() and print():",
    'svamp_prompt_cot': "\n\nPlease format the final answer at the end of the response as: The answer is {answer}.",
    'bbh_prompt_cot': "\n\nPlease format the final answer at the end of the response as: The answer is {answer}.",
    'math_prompt_cot': "\n\nPlease format the final answer at the end of the response as: The answer is {answer}.",
    'mmedbench_en_prompt_cot': "\n\nPlease answer with option letter directly, do not output other infomation.",
    'mmlu_prompt_cot': "\n\nPlease answer with option letter directly, do not output other infomation.",
    'logiqa_en_prompt_cot': "\n\nPlease think step by step and give your answer in the end.",
    'commonsense_qa_prompt_cot': "\n\nLet's think step by step. Please format the final answer at the end of the response as: The answer is {answer}.",
    'svamp_100_prompt_cot': "\n\nPlease format the final answer at the end of the response as: The answer is {answer}.",
    'gsm8k': "\n\nPlease format the final answer at the end of the response as: The answer is {answer}.",
    'mmedbench_en_cot': "\n\nPlease format the final answer at the end of the response as: The answer is {answer}.",
    'mmedbench_zh_cot': "\n\n请在回答的最后用以下格式回答：答案为{answer}。",
    'PLE_Pharmacy_cot': "\n\n请在回答的最后用以下格式回答：答案为{answer}。",
    'PLE_TCM_cot': "\n\n请在回答的最后用以下格式回答：答案为{answer}。",
    'math': "\n\nPlease format the final answer at the end of the response as:  The answer is {answer}.",
    'math_500': "\n\nPlease format the final answer at the end of the response as:  The answer is {answer}.",
    'winogrande': "\n\nPlease answer with option letter directly, do not output other infomation."
}

def get_fewshot_examples(dataset_name, num_samples=3):
    dataset = dataset_name.split("_")[0]
    mapping = {
        "bbh": "ming/eval/fewshot_examples/bbh_prompt_cot/example.txt",
        "svamp": "ming/eval/fewshot_examples/svamp_prompt_cot/example.txt",
        "math": "ming/eval/fewshot_examples/math_prompt_cot/example.txt",
        "commonsense": "ming/eval/fewshot_examples/commonsense_qa_prompt_cot/example.txt",
        "logiqa": "ming/eval/fewshot_examples/logiqa_en_prompt_cot/example.txt",
        "mmlu": "ming/eval/fewshot_examples/mmlu_prompt_cot/example.txt",
        "mmedbench": "ming/eval/fewshot_examples/mmedbench_en_prompt_cot/example.txt"
    }
    samples = {
        "mmlu": 3,
        "commonsense": 1,
        "logiqa": 10,
        "mmedbench": 10,
        "svamp": 10
    }
    if dataset in mapping:
        print("Loading few shot examples...")
        few_shot_prompt = open(mapping[dataset], 'r').read()
        each_examples = few_shot_prompt.split("Problem: ")
        few_shot_prompt = "Problem: ".join(each_examples[:samples[dataset]])
        print(few_shot_prompt, flush=True)
    else:
        few_shot_prompt = "Please directly answer with the answer letter.\n"
    return few_shot_prompt
    # pass
    
class CustomDataset:
    def __init__(self, questions, batch_size, task_specific_prompt, dataset_name='default', tokenizer=None,
                 is_base=False, add_few_shot=False, num_samples=3):
        self.questions = questions
        self.batch_size = batch_size
        self.size = len(questions)

        # self.conv_mode = conv_mode
        self.task_specific_prompt = task_specific_prompt
        self.dataset_name = dataset_name
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        self.tokenizer = tokenizer
        self.is_base = is_base
        # print(add_few_shot, flush=True)
        if add_few_shot: 
            self.few_shot_samples = get_fewshot_examples(self.dataset_name, num_samples)
        else:
            self.few_shot_samples = ""
        if 'zh' in dataset_name:
            self.cot_prompt = "\n答案为："
        else:
            self.cot_prompt = "\nThe answer is "
            
    def __getitem__(self, index):
        bz = self.batch_size
        items = []
        # return question, ansewr, additional info
        # questions = []
        # prompts = []
        # answers = []
        # additional_infos = []
        for i in range(index*bz, (index+1)*bz):
            if i < self.size:
                # conv = self.conv.copy()
                if self.dataset_name.endswith("plus"):
                    question = self.questions[i]['prompt']
                    if self.is_base: 
                        if self.few_shot_samples == "":
                            prompt = self.few_shot_samples + "Problem: " + question + "\n\nAnswer: The answer is "
                        else:
                            prompt = self.few_shot_samples + question + "\n</problem>\n\n<AnswerText> Let's think step by step. "
                    else:
                        prompt = question
                    alpaca_item = {
                        "id": f"{self.dataset_name}_{i}",
                        "conversations": [
                            {"from": "human", "value": prompt},
                        ],
                        "eval": self.questions[i]['task_id']
                    }
                    item = AlpacaTaskItem(alpaca_item, self.task_specific_prompt)
                    # answers.append(None)
                    items.append(item)
                else:
                    line = self.questions[i]
                    question = line['conversations'][0]['value']
                    # questions.append(question)
                    if self.is_base: 
                        if self.few_shot_samples == "":
                            question = self.few_shot_samples + "Problem: " + question + "\n\nAnswer: "
                        else:
                            question = self.few_shot_samples  + "Problem: " + question + "\n\nAnswer:"
                    else:
                        if self.few_shot_samples != "":
                            question = self.few_shot_samples + "Problem: " + question + "\n\nAnswer: "
                        else:
                            question = question 

                    alpaca_item = line 
                    alpaca_item['conversations'][0]['value'] = question
                    item = AlpacaTaskItem(alpaca_item, self.task_specific_prompt)
                    items.append(item)
                    # answers.append(line['conversations'][1]['value'] if len(line['conversations']) > 1 else None)
                    # additional_infos.append(line['eval'] if 'eval' in line else None)

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return items

    def __len__(self):
        return math.ceil(len(self.questions) / self.batch_size)

    def __iter__(self):
        # 返回迭代器对象本身
        return self
    
    def __next__(self):
        if self.index < len(self.questions):
            # 返回下一个值并更新索引
            item = self.questions[self.index]
            self.index += 1
            return item
        else:
            # 没有更多元素时抛出StopIteration异常
            raise StopIteration
        

def get_loss(logits, labels, attention_mask, vocab_size):
    from torch.nn import CrossEntropyLoss
    labels = labels.masked_fill(~attention_mask, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    B, N, C = shift_logits.shape
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    # this loss is [-1, ], we need to reshape it to [B, N]
    loss = loss.reshape(B, N)
    # we must know that some positions are 0-loss because of ignore_index, we need to ignore these
    loss_sum = loss.sum(dim=-1)
    loss_actual_position = torch.not_equal(loss, 0).sum(dim=-1)
    loss = loss_sum / loss_actual_position  # [B, ]
    return loss


def set_tokenizer(tokenizer):
    tokenizer.padding_side = 'left'
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    return tokenizer



def infer_answer(tokenizer, model, prompts, outputs, lora_request=None, choices_word=None, cot_prompt="\nThe answer is "):
    if len(prompts) != len(outputs):
        # copy
        number = len(outputs) // len(prompts)
        prompts = [x for x in prompts for _ in range(number)]

    cot_prompts = [(prompt + output + f"{' ' if output.strip().endswith('.') else '. '}{cot_prompt}") for prompt, output in zip(prompts, outputs)]
    
    def add_choice_words(choices_word):
        choice_ids = [tokenizer.encode(x)[0] for x in choices_word]
        return choice_ids
    
    if choices_word is not None:
        logits_processor = LogitsProcessorList()
        logits_processor.append(LogitBiasProcess(add_choice_words(choices_word)))
        new_tokens = 1
    else:
        logits_processor = None 
        new_tokens = 50
    
    answer_outputs = vllm_clean_generate(
        model,
        cot_prompts,
        lora_request=lora_request,
        wrap_chat=False,
        temperature=0,
        logits_processors=logits_processor,
        max_tokens=new_tokens,
        n=1 
    )
    answer_outputs = [x[0] for x in answer_outputs]
    cur_outputs = [f"{output}{' ' if output.strip().endswith('.') else '. '}{cot_prompt}{answer_output}." for output, answer_output in zip(outputs, answer_outputs)]
    return cur_outputs   




