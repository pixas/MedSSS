from abc import ABC, abstractmethod
import json 
from Evol_Instruct.models.vllm_support import VLLMServer
from Evol_Instruct.utils.utils import AlpacaTaskItem
from transformers import LogitsProcessorList
from Evol_Instruct.utils.utils import LogitBiasProcess


class Solver(ABC):
    def __init__(self, server: VLLMServer, save_file, choices_word=None, **kwargs):
        """
        初始化 Solver 类，接受一个模型和一个分词器。
        :param model: 用于推理的大模型实例
        :param tokenizer: 用于分词的分词器实例
        """
        self.server = server
        # self.model = model
        # self.tokenizer = tokenizer
        # self.lora_request = lora_request
        
        self.save_file = save_file
        self.responses = []  # 用于保存生成的回复
        self.choices_word = choices_word
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def generate_response(self, item: list[AlpacaTaskItem], **kwargs):
        """
        抽象方法，用于生成回复。子类需要实现具体的推理逻辑。
        :param items: 输入一个batch的问题，包括问题的其他元信息
        :return: 生成的回复
        """
        pass

    def save_response(self, items: list[AlpacaTaskItem]):
        """
        保存生成的回复。
        :param response: 要保存的回复内容
        """
        for item in items:
            self.save_file.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        
        self.save_file.flush()
    
    def infer_answer(self, prompts, outputs, choices_word=None, cot_prompt="\nThe answer is ", add_previous_output=True):
        """
        从生成的回复中推断答案。
        :param response: 生成的回复
        :return: 推断的答案
        """
        if len(prompts) != len(outputs):
            # copy
            number = len(outputs) // len(prompts)
            prompts = [x for x in prompts for _ in range(number)]
        
        cot_prompts = [(prompt + output + f"{' ' if output.strip().endswith('.') else '. '}{cot_prompt}") for prompt, output in zip(prompts, outputs)]
    
        def add_choice_words(choices_word):
            choice_ids = [self.server.tokenizer.encode(x)[0] for x in choices_word]
            return choice_ids
        
        if choices_word is not None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(LogitBiasProcess(add_choice_words(choices_word)))
            new_tokens = 1 
            if getattr(self, 'infer_answer_max_tokens', None) is not None:
                logits_processor = None
        else:
            logits_processor = None 
            new_tokens = 128
        
        answer_outputs = self.server(cot_prompts,
                                     wrap_chat=False,
                                     system='You are a helpful assistant.',
                                     temperature=0,
                                     logits_processors=logits_processor,
                                     max_tokens=new_tokens,
                                     n=1)
        answer_outputs = [x[0].strip() for x in answer_outputs]
        if add_previous_output:
            cur_outputs = [f"{output}{' ' if output.strip().endswith('.') else '. '}{cot_prompt}{answer_output}." for output, answer_output in zip(outputs, answer_outputs)]
        else:
            cur_outputs = [f'{cot_prompt}{answer_output}.' for answer_output in answer_outputs]
        return cur_outputs   