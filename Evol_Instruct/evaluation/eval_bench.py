import json
from math import e
import pathlib

from tqdm import tqdm
from Evol_Instruct import client
import re 
from scipy.special import comb
import argparse
import datetime
from datetime import datetime
import math 
import numpy as np
from Evol_Instruct.MCTS.utils import extract_template
from Evol_Instruct.utils.utils import compute_weighted_values
from openai import OpenAI
import os 


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
    
def extract_answers(text):
    try:
        result = re.findall(r'<answer.*?>(.*?)</answer>', text, flags=re.DOTALL)[-1]
        return [result.strip("\n").strip()]
    except:
        return None
def process_medcalc_pred(pred_answer):

    however_index = pred_answer.find("However")
    if however_index != -1:
        pred_answer = pred_answer[:however_index].strip().strip(".")
    left_bracket_index = pred_answer.find("(")
    if left_bracket_index != -1:
        pred_answer = pred_answer[:left_bracket_index].strip().strip(".")
    return pred_answer

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
    # if "<answer>" in s and "</answer>" in s:
    #     possible_output = extract_answers(s)
    if "<think>" in s and "</think>" in s:
        answer_content = s.split("</think>")[-1].strip().split("\nThe answer is")[0].strip()
        extracted_answer = answer_content.split("Answer: ")[-1].strip()
        return [extracted_answer]
        # if possible_output
    # match1 = re.findall(r'the answer is (.+)\.', s, )
    patterns = [
        rf"{prefix}([^\n]*)(?:\.|\n|$)",
        r"The correct answer is([^\n]*)(?:\.|\n|$)",
        r"Final Answer[:\s]*([^\n]*)(?:\.|\n|$)",
        r"Final answer is([^\n]*)(?:\.|\n|$)"
    ]
    matches = []
    match_by_pat = {}
    for pat in patterns:
        match_by_pat[pat] = []
        for m in re.finditer(pat, s, re.IGNORECASE):
            match_by_pat[pat].append(m.group(1))
            # matches.append(m.group(1))
        if match_by_pat[pat]:
            return [match_by_pat[pat][-1].strip(":").strip().strip(".").strip("*")]

    return [s.split("\n")[-1].strip().strip(".")]
    
def acc_score(answer, pred, text_answer=None):
    if pred is None:
        acc = 0
    if len(answer) == 1:
        # str.upper()
        clean_pred = pred.replace("'", "").replace('"', '').upper()
        if len(clean_pred) == 0:
            acc = 0
        else:
            if text_answer is not None:
                acc = int(answer[0] in clean_pred or text_answer in clean_pred.lower())
            else:
                acc = int(answer[0] in clean_pred[0])
    else:
        clean_answer = answer.lower()
        clean_pred = pred.lower()
        acc = int(clean_answer in clean_pred)
    return acc

    
def multiplechoice_acc(line, eval_meds3=False):
    pred = line['text']
    if eval_meds3:
        pred = pred.rsplit("\n", 1)[1]
    ground_truth = line['additional_info']['answer']
    text_answer = line.get('text_answer', None)
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
        
        final_answer = extract_pred[0].strip('`')
        if len(final_answer) == 0:
            acc = 0 
        else:
            final_answer = final_answer[0]
            acc = acc_score(ground_truth, final_answer, text_answer)
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


def extract_medcalc(extracted_answer, calid):

    calid = int(calid)
  
    
    
    if calid in [13, 68]:
        # Output Type: date
        match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = match.group(3)
            answer = f"{month:02}/{day:02}/{year}"
        else:
            answer = "N/A"

    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
        # ground_truth = f"({match.group(1)}, {match.group(3)})"
        extracted_answer = extracted_answer.replace("[", "(").replace("]", ")").replace("'", "").replace('"', "")
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f"({weeks}, {days})"
        else:
            answer = "N/A"
    elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
        # Output Type: integer A
        match = re.search(r"(\d+) out of", extracted_answer)
        if match: # cases like "3 out of 5"
            answer = match.group(1)
        else:
            match = re.search(r"-?\d+(, ?-?\d+)+", extracted_answer)
            if match: # cases like "3, 4, 5"
                answer = str(len(match.group(0).split(",")))
            else:
                # match = re.findall(r"(?<!-)\d+", extracted_answer)
                match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                # match = re.findall(r"-?\d+", extracted_answer)
                if len(match) > 0: # find the last integer
                    answer = match[-1][0]
                    # answer = match[-1].lstrip("0")
                else:
                    answer = "N/A"
    elif calid in [2,  3,  5,  6,  7,  8,  9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:
        # Output Type: decimal
        match = re.search(r"str\((.*)\)", extracted_answer)
        if match: # cases like "str(round((140 * (3.15 - 136) / 1400) * 72.36)"
            expression = match.group(1).replace("^", "**").replace("is odd", "% 2 == 1").replace("is even", "% 2 == 0").replace("sqrt", "math.sqrt").replace(".math", "").replace("weight", "").replace("height", "").replace("mg/dl", "").replace("g/dl", "").replace("mmol/L", "").replace("kg", "").replace("g", "").replace("mEq/L", "")
            expression = expression.split('#')[0] # cases like round(45.5 * 166 - 45.3 + 0.4 * (75 - (45.5 * 166 - 45.3))))) # Calculation: ...
            if expression.count('(') > expression.count(')'): # add missing ')
                expression += ')' * (expression.count('(') - expression.count(')'))
            elif expression.count(')') > expression.count('('): # add missing (
                expression = '(' * (expression.count(')') - expression.count('(')) + expression
            try:
                answer = eval(expression, {"__builtins__": None}, {"min": min, "pow": pow, "round": round, "abs": abs, "int": int, "float": float, "math": math, "np": np, "numpy": np})
            except:
                print(f"Error in evaluating expression: {expression}")
                answer = "N/A"
        else:
            match = re.search(r"(-?\d+(\.\d+)?)\s*mL/min/1.73", extracted_answer)
            if match: # cases like "8.1 mL/min/1.73 m\u00b2"
                answer = eval(match.group(1))
            else:
                match = re.findall(r"(-?\d+(\.\d+)?)\%", extracted_answer)
                if len(match) > 0: # cases like "53.1%"
                    answer = eval(match[-1][0]) / 100
                else:
                    match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                    if len(match) > 0: # cases like "8.1 mL/min/1.73 m\u00b2" or "11.1"
                        try:
                            answer = eval(match[-1][0].lstrip("0"))
                        except:
                            answer = "N/A"
                    else:
                        answer = "N/A"
        if answer != "N/A":
            answer = str(answer)          
 
    return answer 
    
def medsins_acc(line, eval_meds3=False):
    pred = line['text']
    if not pred.endswith("."):
        pred = pred + "."
    if eval_meds3:
        pred = pred.rsplit("\n", 1)[1]
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
    if "The diagnosis result" in pred:
        pred_answer = [pred.split('\n')[0].split("The diagnosis result is ")[-1].strip().strip(".")]
    else:
        pred_answer = extract_answer_content(pred)
    # if "The answer is " not in pred and "the answer is " not in pred:
    #     pred_answer = [pred.split("\n")[-1].strip().strip(".")]
    # else:
    #     if "The diagnosis result" in pred:
    #         # medsins llama3
    #         pred_answer = [pred.split('\n')[0].split("The diagnosis result is ")[-1].strip().strip(".")]
    #     else:
    #         pred_answer = extract_answer_content(pred)
    extract_all_pred = []
    for x in answer_text:
        if x is None:
            extract_all_pred.append([x])
        else:
            extract_all_pred.append(extract_answer_content(x) if "the answer is" in x.lower() else [x])

    
    if pred_answer is not None:

        pred_answer = pred_answer[0].strip('`')
        
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




def check_medcalc_correctness(answer: str, ground_truth, calid, upper_limit, lower_limit):

    calid = int(calid)

    if calid in [13, 68]:
        # Output Type: date
        try:
            if datetime.strptime(answer, "%m/%d/%Y").strftime("%-m/%-d/%Y") == datetime.strptime(ground_truth, "%m/%d/%Y").strftime("%-m/%-d/%Y"):
                correctness = 1
            else:
                correctness = 0
        except:
            correctness = 0
    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", ground_truth)
        ground_truth = f"({match.group(1)}, {match.group(3)})"
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f"({weeks}, {days})"
            try:
                if eval(answer) == eval(ground_truth):
                    correctness = 1
                else:
                    correctness = 0
            except:
                correctness = 0
        else:
            correctness = 0
    elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
        # Output Type: integer A
        try:
            answer = round(eval(answer))
            if answer == eval(ground_truth):
                correctness = 1
            else:
                correctness = 0
        except:
            correctness = 0
    elif calid in [2,  3,  5,  6,  7,  8,  9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:
        # Output Type: decimal
        try:
            answer = eval(answer)
            if answer >= eval(lower_limit) and answer <= eval(upper_limit):
                correctness = 1
            else:
                correctness = 0
        except:
            correctness = 0
    else:
        raise ValueError(f"Unknown calculator ID: {calid}")
    return correctness

def medcalc_acc(line, eval_meds3=False):
    pred = line['text']
    if not pred.endswith("."):
        pred = pred + "."
    if eval_meds3:
        pred = pred.rsplit("\n", 1)[1]
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
    if "The diagnosis result" in pred:
        pred_answer = [pred.split('\n')[0].split("The diagnosis result is ")[-1].strip().strip(".")]
    else:
        pred_answer = extract_answer_content(pred)
    # if "The answer is " not in pred and "the answer is " not in pred:
    #     pred_answer = [pred.split("\n")[-1].strip().strip(".")]
    # else:
    #     if "The diagnosis result" in pred:
    #         # medsins llama3
    #         pred_answer = [pred.split('\n')[0].split("The diagnosis result is ")[-1].strip().strip(".")]
    #     else:
    #         pred_answer = extract_answer_content(pred)
    extract_all_pred = []
    for x in answer_text:
        if x is None:
            extract_all_pred.append([x])
        else:
            extract_all_pred.append(extract_answer_content(x) if "the answer is" in x.lower() else [x])


    calculation_id = line['additional_info']['Calculator ID']
    if pred_answer is not None:

        pred_answer: str = pred_answer[0].strip('`')
        pred_answer = process_medcalc_pred(pred_answer)
        extracted_answer = extract_medcalc(pred_answer, calculation_id)
        acc = check_medcalc_correctness(extracted_answer, ground_truth, calculation_id, line['additional_info']["Upper Limit"], line['additional_info']["Lower Limit"])
        # acc = acc_score(ground_truth, pred_answer)
    else:
        acc = 0
    all_pred = []
    for pred in extract_all_pred:
        if pred is not None:
            pred = pred[0]
            pred = process_medcalc_pred(pred)
            extracted_answer = extract_medcalc(pred, calculation_id)
            acc = check_medcalc_correctness(extracted_answer, ground_truth, calculation_id, line['additional_info']["Upper Limit"], line['additional_info']["Lower Limit"])
            all_pred.append(acc)
    
    total_num = len(all_pred)
    correct_num = sum(all_pred)
    if total_num != 0:
        for k in range(1, 3):
            pass_at_k_metric.append(pass_at_k(total_num, correct_num, k))
    return pred_answer, acc, pass_at_k_metric

def rdc_acc(line, eval_meds3=False):
    pred = line['text']
    if not pred.endswith("."):
        pred = pred + "."
    if eval_meds3:
        pred = pred.rsplit("\n", 1)[1]
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
    if "The diagnosis result" in pred:
        pred_answer = [pred.split('\n')[0].split("The diagnosis result is ")[-1].strip().strip(".")]
    else:
        pred_answer = extract_answer_content(pred)
    # if "The answer is " not in pred and "the answer is " not in pred:
    #     pred_answer = [pred.split("\n")[-1].strip().strip(".")]
    # else:
    #     if "The diagnosis result" in pred:
    #         # medsins llama3
    #         pred_answer = [pred.split('\n')[0].split("The diagnosis result is ")[-1].strip().strip(".")]
    #     else:
    #         pred_answer = extract_answer_content(pred)
    extract_all_pred = []
    for x in answer_text:
        if x is None:
            extract_all_pred.append([x])
        else:
            extract_all_pred.append(extract_answer_content(x) if "the answer is" in x.lower() else [x])

    def diagnosis_acc(ground_truth, pred):
        
        ground_truth = ground_truth.lower()
        pred = pred.lower()
        pred = pred.strip("\"").strip("'").strip("`")
        if ground_truth in pred or pred in ground_truth:
            return 1
        else:
            return 0
    
    def gpt_acc(ground_truth, pred):
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        # os.environ['http_proxy'] = os.environ['https_proxy'] = os.environ['GPT_PROXY']
        message = [
            {"role": "user", "content": "Given a reference ground truth and a model's predicted answer, judge whether the predicted answer is correct or not. If the predicted answer is correct, return 1, otherwise return 0.\n\nGround truth: " + ground_truth + "\nPredicted answer: " + pred + "\nYour judgement: "},
        ]
        response = client.chat.completions.create(
            messages=message,
            model="gpt-3.5-turbo",
            max_tokens=10,
            temperature=0,
        )
        verdict = response.choices[0].message.content
        return 1 if "1" in verdict else 0
    
    if pred_answer is not None:

        pred_answer = pred_answer[0].strip('`')
        
        acc = diagnosis_acc(ground_truth, pred_answer)
        if acc == 0:
            acc = gpt_acc(ground_truth, pred_answer)
        
    else:
        acc = 0
    all_pred = []
    for pred in extract_all_pred:
        if pred is not None:
            pred = pred[0]
            acc = diagnosis_acc(ground_truth, pred)
            if acc == 0:
                acc = gpt_acc(ground_truth, pred)
            all_pred.append(acc)
    
    total_num = len(all_pred)
    correct_num = sum(all_pred)
    if total_num != 0:
        for k in range(1, 3):
            pass_at_k_metric.append(pass_at_k(total_num, correct_num, k))
    return pred_answer, acc, pass_at_k_metric, all_pred
    pass

score_mapping = {
    "MedQA": multiplechoice_acc,
    "MedMCQA": multiplechoice_acc,
    'med_mmlu': multiplechoice_acc,
    'pubmedqa': multiplechoice_acc,
    "bioasq": multiplechoice_acc,
    "pubhealth": multiplechoice_acc,
    "biomrc": multiplechoice_acc,
    "healthfact": medsins_acc,
    "drugdose": medsins_acc,
    "ddxplus": medsins_acc,
    "seer": medsins_acc,
    "medexpert": multiplechoice_acc,
    "medcalc": medcalc_acc,
    "rdc": rdc_acc,
    'rds': rdc_acc
}



class Scorer:
    def __init__(self, input_file: str, output_file: str, eval_mcts_method: str = None, eval_meds3=False):

        self.input_data = client.read(input_file)
        self.dataset = self.obtain_dataset(input_file)
        self.score_func = score_mapping[self.dataset]
        score_file = pathlib.Path(output_file)
        self.score_cache = score_file.parent / "cache.jsonl"
        self.score_cache_file = open(self.score_cache, "w")
        self.score_file = score_file.parent / "result.json"
        self.output_file = output_file
        self.eval_mcts_method = eval_mcts_method
        self.eval_meds3 = eval_meds3
    
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
            if self.dataset == 'rdc' or self.dataset == 'rds':
                final_answer, acc, pass_at_k_metric, all_acc = self.score_func(line, self.eval_meds3)
            else:
                all_acc = None
                final_answer, acc, pass_at_k_metric = self.score_func(line, self.eval_meds3)
            
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
                "id": line.get("id", None),
                "acc": acc,
                "pred": final_answer,
                "ground_truth": line['additional_info']['answer'],
                "text_answer": line.get('text_answer', None),
                "pass_at_k": pass_at_k_metric,
                "all_acc": all_acc,
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
    parser.add_argument("--eval_meds3", action='store_true')
    parser.add_argument('--eval_mcts_method', default=None, type=str)
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = args.input_file.replace(".jsonl", "_wrong.json")
    scorer = Scorer(args.input_file, args.output_file, args.eval_mcts_method,
                    eval_meds3=args.eval_meds3)
    
    scorer.score_one()