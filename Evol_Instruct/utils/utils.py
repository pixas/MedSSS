import scipy.stats
import torch 

import numpy as np
from collections import Counter, defaultdict
import json
from petrel_client.client import Client
import io
import os 
from functools import wraps
import math
import re
from contextlib import contextmanager
from transformers import LogitsProcessor



def proxy_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ori_http_proxy = os.environ.get('http_proxy')  # 获取原始的http_proxy值
        ori_https_proxy = os.environ.get("https_proxy")
        os.environ['http_proxy'] = ''  # 在函数执行前将http_proxy设为空字符串
        os.environ['https_proxy'] = ''
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        result = func(*args, **kwargs)  # 执行函数
        os.environ['http_proxy'] = ori_http_proxy if ori_http_proxy is not None else ''  # 函数执行后恢复原始的http_proxy值
        os.environ['https_proxy'] = ori_https_proxy if ori_https_proxy is not None else ''
        os.environ['HTTP_PROXY'] = ori_http_proxy if ori_http_proxy is not None else ''
        os.environ['HTTPS_PROXY'] = ori_https_proxy if ori_https_proxy is not None else ''
        return result
    return wrapper

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class CephOSSClient:
    
    @proxy_decorator
    def __init__(self, conf_path: str = "~/petreloss.conf") -> None:
        self.client = Client(conf_path)
    
    @proxy_decorator
    def read_json(self, json_path, **kwargs):
        if json_path.startswith("s3://"):
            cur_bytes = self.client.get(json_path)
            if cur_bytes != "":
                data = json.loads(cur_bytes, **kwargs)
            else:
                data = []
            # data = json.loads(self.client.get(json_path), **kwargs)
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f, **kwargs)
        return data 

    @proxy_decorator
    def write_json(self, json_data, json_path, **kwargs):
        if json_path.startswith("s3://"):
            if json_data == []:
                self.client.put(json_path, "".encode("utf-8"))
            else:
                self.client.put(json_path, json.dumps(json_data, **kwargs).encode("utf-8"))
        else:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, **kwargs)
        return 1

    @proxy_decorator
    def read_jsonl(self, jsonl_path):
        if jsonl_path.startswith("s3://"):
            bytes = self.client.get(jsonl_path)
            data = bytes.decode('utf-8').split("\n")
            data = [json.loads(x) for x in data if x != ""]
        else:
            data = [json.loads(x) for x in open(jsonl_path, encoding='utf-8', mode='r')]
        return data 
    
    @proxy_decorator
    def write_jsonl(self, jsonl_data, jsonl_path, **kwargs):
        if jsonl_path.startswith("s3://"):
            if jsonl_data == []:
                self.client.put(jsonl_path, "".encode("utf-8"))
                return 1
            if isinstance(jsonl_data, list):
                large_bytes = "\n".join([json.dumps(x, ensure_ascii=False) for x in jsonl_data]).encode("utf-8")
            else:
                large_bytes = (json.dumps(x, ensure_ascii=False) + "\n").encode('utf-8')
            with io.BytesIO(large_bytes) as f:
                self.client.put(jsonl_path, f)
        else:
            with open(jsonl_path, 'w', **kwargs) as f:
                for x in jsonl_data:
                    f.write(json.dumps(x, ensure_ascii=False))
                    f.write("\n")
        return 1

    @proxy_decorator
    def read_txt(self, txt_path):
        if txt_path.startswith("s3://"):
            bytes = self.client.get(txt_path)
            data = bytes.decode('utf-8')
        else:
            with open(txt_path, 'r', encoding='utf-8') as f:
                data = f.read()
        return data 

    @proxy_decorator
    def write_text(self, txt_data, txt_path, mode='w'):
        if txt_path.startswith("s3://"):
            large_bytes = txt_data.encode("utf-8")
            with io.BytesIO(large_bytes) as f:
                self.client.put(txt_path, f)
        else:
            with open(txt_path, mode, encoding='utf-8') as f:
                f.write(txt_data)
        return 1
    
    @proxy_decorator
    def save_checkpoint(self, data, path, **kwargs):
        if "s3://" not in path:
            assert os.path.exists(path), f'No such file: {path}'
            torch.save(data, path, **kwargs)
        else:
            with io.BytesIO() as f:
                torch.save(data, f, **kwargs)
                self.client.put(f.getvalue(), path)
        return 1 

    @proxy_decorator
    def load_checkpoint(self, path, map_location=None, **kwargs):
        if "s3://" not in path:
            assert os.path.exists(path), f'No such file: {path}'
            return torch.load(path, map_location=map_location, **kwargs)
        else:
            file_bytes = self.client.get(path)
            buffer = io.BytesIO(file_bytes)
            res = torch.load(buffer, map_location=map_location, **kwargs)
            return res
    
    @proxy_decorator
    def exists(self, file_path):
        if "s3://" not in file_path:
            return os.path.exists(file_path)
        else:
            return self.client.contains(file_path)
    
    @proxy_decorator
    def remove(self, file_path):
        if "s3://" not in file_path:
            return os.remove(file_path)
        else:
            return self.client.delete(file_path)
    
    @proxy_decorator
    def read_csv(self, path):
        if "s3://" in path:
            bytes = self.client.get(path)
            data = bytes.decode('utf-8').split("\n")
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = f.readlines()
        return data

    def read(self, path: str):
        s3_prefix = "s3://syj_test"
        local_prefix = "/mnt/petrelfs/jiangshuyang.p/oss"
        if "local_prefix" in path:
            path = path.replace(local_prefix, s3_prefix)
        mapping_processing = {
            "csv": self.read_csv,
            "json": self.read_json,
            "jsonl": self.read_jsonl,
            "txt": self.read_txt,
            "log": self.read_txt
        }
        suffix = path.split(".")[-1]
        try:
            return mapping_processing[suffix](path)
        except:
            s3_prefix = "s3://syj_test"
            local_prefix = "/mnt/petrelfs/jiangshuyang.p/oss"
            path = path.replace(local_prefix, s3_prefix)
            return mapping_processing[suffix](path)
    
    def write(self, data, path: str, **kwargs):
        mapping_processing = {
            "csv": self.write_text,
            "json": self.write_json,
            "jsonl": self.write_jsonl,
            "txt": self.write_text,
            "log": self.write_text
        }
        suffix = path.split(".")[-1]
        try:
            return mapping_processing[suffix](data, path, **kwargs)
        except Exception as e:
            print(e)
            s3_prefix = "s3://syj_test"
            local_prefix = "/mnt/petrelfs/jiangshuyang.p/oss"
            path = path.replace(local_prefix, s3_prefix)
            return mapping_processing[suffix](data, path, **kwargs)

    @proxy_decorator
    def listdir(self, path):
        if "s3://" in path:
            output = [x for x in list(self.client.list(path)) if x != ""]
            return output
        else:
            return os.listdir(path)


client = CephOSSClient("~/petreloss.conf")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def extract_backticks(s):
# 正则表达式模式，匹配三个反引号开始和结束的字符串
    pattern = r'```(.*?)```'
    
    # 使用re.findall来查找所有匹配项
    matches = re.findall(pattern, s, re.DOTALL)
    
    # 返回所有匹配的内容
    return [match.strip('\n') for match in matches]
    


def select_standard_output(text_list, standard_list, is_exist=True):
    ret_text = None 
    for each in text_list:
        if any(x in each.lower() if is_exist else x not in each.lower() for x in standard_list):
            continue 
        ret_text = each 
    return ret_text


def extract_answer(s, prefix='The answer is'):
    if not s.endswith("."):
        s = s + "."
    # match1 = re.findall(r'the answer is (.+)\.', s, )
    match2 = re.findall(prefix + r' (.+)\.', s, )
    if match2:
        return match2[-1][0]
    else:
        return None

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



class AlpacaTaskItem:
    def __init__(self, input_item, task_specific_prompt=""):
        self.id = input_item['id']
        self.question = input_item['conversations'][0]['value']
        if len(input_item['conversations']) == 1:
            self.original_response = None
        else:
            self.original_response = input_item['conversations'][1]['value']
        self.additional_info = input_item.get("eval", None)
        self.answer = self.additional_info['answer'] if self.additional_info is not None else None
        self.prompt = self.question + task_specific_prompt
    
    def to_dict(self):
        output = {k: v for k, v in self.__dict__.items()}
        return output
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        

class LogitBiasProcess(LogitsProcessor):
    def __init__(self, activate_token_list: list[int] = None, activate_scale=100):
        self.activate_token_list = activate_token_list
        self.activate_scale=activate_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # logger.info(scores.shape, input_ids.shape)
        for id_ in self.activate_token_list:
            if scores.dim() == 2:
                scores[:, id_] += self.activate_scale
            else:
                scores[id_] += self.activate_scale
        return scores



def compute_weighted_values(only_answer_outputs, values, method: str):
    mapping = {
        "min": lambda x: np.min(x[1:]),
        "mean": lambda x: np.mean(x[1:]),
        "gmean": lambda x: scipy.stats.gmean(x[1:]),
        "prod": lambda x: np.prod(x[1:])
    }
    if method.startswith("prm"):
        method = method.split("prm-")[1]
        value_op = method.split("-")[0]
        if value_op == "vote":
            value_op = lambda x: x[-1]
        else:
            value_op = mapping[value_op]
    else:
        value_op = lambda x: x 
    answer_count = Counter([answer for answer in only_answer_outputs])
    values = [value_op(x) for x in values]
    answer_dict = defaultdict(float)
    for answer, score in zip(only_answer_outputs, values):
        answer_dict[answer] += score
    # print(method)
    if "vote-sum" in method:
        try:
            final_answer = max(answer_dict, key=answer_dict.get)
        except:
            print(only_answer_outputs, values)
            exit(-1)
    elif "vote-mean" in method:
        for answer in answer_dict:
            answer_dict[answer] /= answer_count[answer]
        final_answer = max(answer_dict, key=answer_dict.get)
    elif "max" in method:
        final_answer = max(zip(only_answer_outputs, values), key=lambda x: x[1])[0]
    # if method == 'vote-sum':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         if isinstance(score, list):
    #             score = score[-1]
    #         answer_dict[answer] += score
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'vote-mean':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         if isinstance(score, list):
    #             score = score[-1]
    #         answer_dict[answer] += score
    #     answer_count = Counter([answer for answer in only_answer_outputs])
    #     for answer in answer_dict:
    #         answer_dict[answer] /= answer_count[answer]
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'prm-mean-vote-sum':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         answer_dict[answer] += np.mean(score[1:])
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'prm-min-vote-sum':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         answer_dict[answer] += np.min(score[1:])
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'prm-gmean-vote-sum':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         answer_dict[answer] += scipy.stats.gmean(score[1:])
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'prm-gmean-max':
    #     answer_dict = defaultdict(float)
    #     values = [scipy.stats.gmean(x[1:]) for x in values]
    #     # for answer, score in zip(only_answer_outputs, values):
    #     #     answer_dict[answer] += scipy.stats.gmean(score[1:])
    #     final_answer = max(zip(only_answer_outputs, values), key=lambda x: x[1])[0]
    # elif method == 'prm-gmean-vote-mean':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         answer_dict[answer] += scipy.stats.gmean(score[1:])
    #     answer_count = Counter([answer for answer in only_answer_outputs])
    #     for answer in answer_dict:
    #         answer_dict[answer] /= answer_count[answer]
        
    #     final_answer = max(answer_dict, key=answer_dict.get)
        
    # elif method == 'prm-prod-vote-sum':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         answer_dict[answer] += np.prod(score[1:])
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'prm-vote-sum':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         answer_dict[answer] += score[-1]
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'prm-prod-vote-mean':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         answer_dict[answer] += np.prod(score[1:])
    #     answer_count = Counter([answer for answer in only_answer_outputs])
    #     for answer in answer_dict:
    #         answer_dict[answer] /= answer_count[answer]
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'prm-mean-vote-mean':
    #     answer_dict = defaultdict(float)
    #     for answer, score in zip(only_answer_outputs, values):
    #         answer_dict[answer] += np.mean(score[1:])
    #     answer_count = Counter([answer for answer in only_answer_outputs])
    #     for answer in answer_dict:
    #         answer_dict[answer] /= answer_count[answer]
    #     final_answer = max(answer_dict, key=answer_dict.get)
    # elif method == 'max':
    #     if isinstance(values[0], list):
    #         # score = score[-1]
    #         values = [value[-1] for value in values]
    #     answer_dict = defaultdict(float)
    #     pairs = list(zip(only_answer_outputs, values))
    #     final_answer = max(pairs, key=lambda x: x[1])[0]
    # else:
    #     raise NotImplementedError
    return final_answer, answer_dict

@contextmanager
def proxy_manager():
    # 保存原始的代理环境变量
    original_http_proxy = os.environ.get('http_proxy')
    original_https_proxy = os.environ.get('https_proxy')
    original_http_proxy_upper = os.environ.get('HTTP_PROXY')
    original_https_proxy_upper = os.environ.get('HTTPS_PROXY')
    
    # 清空代理环境变量
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    
    try:
        # 进入上下文，执行代码块
        yield
    finally:
        # 恢复原始的代理环境变量
        if original_http_proxy is not None:
            os.environ['http_proxy'] = original_http_proxy
        else:
            os.environ.pop('http_proxy', None)
        
        if original_https_proxy is not None:
            os.environ['https_proxy'] = original_https_proxy
        else:
            os.environ.pop('https_proxy', None)
        
        if original_http_proxy_upper is not None:
            os.environ['HTTP_PROXY'] = original_http_proxy_upper
        else:
            os.environ.pop('HTTP_PROXY', None)
        
        if original_https_proxy_upper is not None:
            os.environ['HTTPS_PROXY'] = original_https_proxy_upper
        else:
            os.environ.pop('HTTPS_PROXY', None)


@contextmanager
def add_proxy():
    proxy_url = os.environ.get("proxy_url")
    os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = proxy_url
    try:
        yield
    finally:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = ''
        

if __name__ == "__main__":
    # data = client.read_jsonl("s3://syj_test/test.jsonl")
    # print(data, type(data), type(data[0]))
    files = client.listdir("s3://syj_test/datasets/medical_train/llama38b_sc16_vllm_mmed_en_trainfilter-random-iter/")
    print(files)
    