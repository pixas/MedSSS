import json
from math import e
import pathlib

from tqdm import tqdm
from Evol_Instruct import client
import re 
from scipy.special import comb
import argparse

from Evol_Instruct.MCTS.utils import extract_template
from Evol_Instruct.utils.utils import compute_weighted_values


def pass_at_k(total_num, correct_num, k):
    if k > total_num:
        return 0
    if k > total_num - correct_num:
        return 1
    # compute the factor
    # how many possible choices to sample k from correct_num 
    factor = comb(total_num - correct_num, k)
    # how many possible choices to sample k from total_num
    total_factor = comb(total_num, k)
    return 1 - factor / total_factor

def extract_answer_content(s, prefix='The answer is'):
    if not s.endswith("."):
        s = s + "."
    # match1 = re.findall(r'the answer is (.+)\.', s, )
    pattern = r"{}(.*)(?:\.|$)".format(prefix)
    matches = list(re.finditer(pattern, s, re.IGNORECASE))
    
    # 如果有匹配项，返回最后一个匹配项的捕获组
    if matches:
        return [matches[-1].group(1).strip().strip(".")]
    else:
        return None
    
def acc_score(answer, pred):
    if pred is None:
        acc = 0
    if len(answer) == 1:
        # str.upper()
        clean_pred = pred.replace("'", "").replace('"', '').upper()
        if len(clean_pred) == 0:
            acc = 0
        else:
            acc = int(answer[0] in clean_pred[0])
    else:
        clean_answer = answer.lower()
        clean_pred = pred.lower()
        acc = int(clean_answer in clean_pred)
    return acc

    
def multiplechoice_acc(line):
    pred = line['text']
    ground_truth = line['additional_info']['answer']
    try:
        all_answer = line['all_answer']
        if isinstance(all_answer[0], str):
            answer_text = all_answer 
        else:
            answer_text = [x[0] for x in all_answer]
    except:
        answer_text = []
    acc = 0 
    pass_at_k_metric = []

    extract_pred = extract_answer_content(pred)
    extract_all_pred = []
    for x in answer_text:
        if x is None:
            extract_all_pred.append(x)
        else:
            extract_all_pred.append(extract_answer_content(x) if "the answer is" in x.lower() else [x])
    # extract_all_pred = [extract_answer_content(x) if x is not None and "the answer is" in x.lower() else [x] for x in answer_text]
    if extract_pred is not None:
        final_answer = extract_pred[0]
        acc = acc_score(ground_truth, final_answer)
    else:
        final_answer = None
        acc = 0
    
    all_pred = [int(any(ground_truth in str(x) for x in pred)) if pred is not None else 0 for pred in extract_all_pred]
    total_num = len(all_pred)
    correct_num = sum(all_pred)
    if total_num != 0:
        for k in range(1, 3):
            pass_at_k_metric.append(pass_at_k(total_num, correct_num, k))
    return final_answer, acc, pass_at_k_metric
    # compute pass_@_k


    
def medsins_acc(line):
    pred = line['text']
    if not pred.endswith("."):
        pred = pred + "."
    ground_truth = line['additional_info']['answer']
    try:
        all_answer = line['all_answer']
        if isinstance(all_answer[0], str):
            answer_text = all_answer 
        else:
            answer_text = [x[0] for x in all_answer]
    except:
        answer_text = []
    
    acc = 0 
    pass_at_k_metric = []
        
    if "The answer is " not in pred:
        pred_answer = [pred]
    else:
        pred_answer = extract_answer_content(pred)
    extract_all_pred = []
    for x in answer_text:
        if x is None:
            extract_all_pred.append(x)
        else:
            extract_all_pred.append(extract_answer_content(x) if "the answer is" in x.lower() else [x])

    
    if pred_answer is not None:

        pred_answer = pred_answer[0]
        acc = acc_score(ground_truth, pred_answer)
    else:
        acc = 0
    all_pred = []
    for pred in extract_all_pred:
        if pred is not None:
            pred = pred[0]
            all_pred.append(acc_score(ground_truth, pred))
    
    total_num = len(all_pred)
    correct_num = sum(all_pred)
    if total_num != 0:
        for k in range(1, 3):
            pass_at_k_metric.append(pass_at_k(total_num, correct_num, k))
    return pred_answer, acc, pass_at_k_metric
    




score_mapping = {
    "MedQA_cot": multiplechoice_acc,
    "MedMCQA_cot": multiplechoice_acc,
    "MedMCQA_cot_500": multiplechoice_acc,
    'pubmedqa_cot': multiplechoice_acc,
    'med_mmlu_cot': multiplechoice_acc,
    'pubmedqa_c_cot': multiplechoice_acc,
    "bioasq": multiplechoice_acc,
    "pubhealth": multiplechoice_acc,
    "biomrc": multiplechoice_acc,
    "biomrc_500": multiplechoice_acc,
    "medsins_task16": medsins_acc,
    "medsins_task16_500": medsins_acc,
    "medsins_task29": medsins_acc,
    "medsins_task130": medsins_acc,
    "medsins_task130_500": medsins_acc,
    "medsins_task131": medsins_acc,
    "medsins_task131_500": medsins_acc,
}



class Scorer:
    def __init__(self, input_file: str, output_file: str, eval_mcts_method: str = None):

        self.input_data = client.read(input_file)
        self.dataset = self.obtain_dataset(input_file)
        self.score_func = score_mapping[self.dataset]
        score_file = pathlib.Path(output_file)
        self.score_cache = score_file.parent / "cache.jsonl"
        self.score_cache_file = open(self.score_cache, "w")
        self.score_file = score_file.parent / "result.json"
        self.output_file = output_file
        self.eval_mcts_method = eval_mcts_method
    
    def obtain_dataset(self, input_file):
        dataset_name = ""
        i = 1
        while dataset_name not in score_mapping:
            dataset_name = input_file.split("/")[-i].replace(".jsonl", "")
            i += 1
        return dataset_name 
    
    def score_one(self):
        acc_list = []
        pass_at_k_list = []
        wrong_idx = []
        for line in tqdm(self.input_data, total=len(self.input_data)):
            final_answer, acc, pass_at_k_metric = self.score_func(line)
            
            if self.eval_mcts_method is not None:
                only_answer_outputs = []
                values = []
                for last_sentence, value in line['all_answer']:
                    if last_sentence is None:
                        continue
                    if "the answer is" in last_sentence.lower():
                        try:
                            only_answer_outputs.append(extract_template(last_sentence, 'answer').lower().replace("'", "").replace('"', ''))
                            values.append(value)
                        except:
                            print(last_sentence)
                            
                    else:
                        # if len(last_sentence) == 1:
                        only_answer_outputs.append(last_sentence)
                        values.append(value)
                        
                        
                
                final_answer, answer_dict = compute_weighted_values(only_answer_outputs, values, self.eval_mcts_method)
                acc = acc_score( line['additional_info']['answer'],final_answer)

            cache_item = {
                "id": getattr(line, 'id', None),
                "acc": acc,
                "pred": final_answer,
                "ground_truth": line['additional_info']['answer'],
                "pass_at_k": pass_at_k_metric
            }
            self.score_cache_file.write(json.dumps(cache_item) + "\n")
            acc_list.append(acc)
            pass_at_k_list.append(pass_at_k_metric)
            if acc == 0:
                wrong_idx.append(line)
        acc = sum(acc_list) / len(acc_list)
        pass_at_k = [sum(x) / len(x) for x in zip(*pass_at_k_list)]
        
        result_term = {
            "dataset": self.dataset,
            "acc": acc,
            "pass_at_k": pass_at_k
        }    
        print(result_term)
        # client.write()
        with open(self.score_file, 'w') as f:
            json.dump(result_term, f)
        
        with open(self.output_file, 'w') as f:
            json.dump(wrong_idx, f, ensure_ascii=False, indent=2)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=False, default=None)
    parser.add_argument("--test_num", type=int, default=-1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument('--eval_mcts_method', default=None, type=str)
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = args.input_file.replace(".jsonl", "_wrong.json")
    scorer = Scorer(args.input_file, args.output_file, args.eval_mcts_method)
    
    scorer.score_one()