import os 
import json 

import argparse 
from tqdm import tqdm, trange
from subprocess import PIPE, Popen, TimeoutExpired
import tempfile
import re 
from pathlib import Path
import Evol_Instruct.evaluation.math_utils as math_utils
from collections import Counter
from sympy import sympify
try:
    from rouge import Rouge
except ImportError:
    import subprocess
    import sys
    
    # 使用 subprocess 执行 pip 安装
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge"])
    
    # 重新尝试导入
    try:
        from rouge import Rouge
    except ImportError:
        print("please install rouge manually")
        sys.exit(1)
        
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text

def add_thousands_separator(s: str) -> str:
    # 处理负数的情况
    is_negative = s.startswith('-')
    if is_negative:
        s = s[1:]

    # 分割整数部分和小数部分
    if '.' in s:
        integer_part, decimal_part = s.split('.')
    else:
        integer_part, decimal_part = s, None

    # 反转整数部分，加入千分位符号后再反转回来
    integer_part = integer_part[::-1]
    integer_part_with_commas = ','.join([integer_part[i:i+3] for i in range(0, len(integer_part), 3)])
    integer_part_with_commas = integer_part_with_commas[::-1]

    # 组合整数部分和小数部分
    if decimal_part:
        result = integer_part_with_commas + '.' + decimal_part
    else:
        result = integer_part_with_commas

    # 如果是负数，加上负号
    if is_negative:
        result = '-' + result

    return result


def normalize_frac(x):
    # Pattern to match \frac{a}{b}
    pattern = r'\\frac\{([^\}]+)\}\{([^\}]+)\}'
    
    # Search for the pattern in the input string
    match = re.search(pattern, x)
    
    # If a match is found, extract 'a' and 'b'
    if match:
        a = match.group(1)  # Numerator
        b = match.group(2)  # Denominator
        
        # Convert to a simplified form, if necessary
        # For demonstration, just return the extracted parts
        return a, b
    else:
        # import pdb 
        # pdb.set_trace()
        return None

def normalize_dfrac(x):
    pattern = r'\\dfrac\{([^\}]+)\}\{([^\}]+)\}'
    
    # Search for the pattern in the input string
    match = re.search(pattern, x)
    
    # If a match is found, extract 'a' and 'b'
    if match:
        a = match.group(1)  # Numerator
        b = match.group(2)  # Denominator
        
        # Convert to a simplified form, if necessary
        # For demonstration, just return the extracted parts
        return a, b
    else:
        # import pdb 
        # pdb.set_trace()
        return None

def normalize(x):
    if "\\frac" in x and normalize_frac(x):
        a, b = normalize_frac(x)
        try:
            a = float(a)
            b = float(b)
            return a / b
        except:
            return x
        
    elif "\\dfrac" in x and normalize_dfrac(x):
        a, b = normalize_dfrac(x)
        try:
            a = float(a)
            b = float(b)
            return a / b
        except:
            return x
    else:
        try:
            x = sympify(x).evalf()
            return float(x)
        except:
            return x

def acc(pred, target):
    return 1 if pred == target else 0

def rouge(pred, target):
    # compute rouge-1, rouge-2, rouge-l
    pass

def extract_bbox_content(s):
    contents = []
    i = 0
    while i < len(s):
        if s[i:i+7] == '\\boxed{':
            depth = 1
            start = i + 7
            i += 7
            while i < len(s) and depth > 0:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        contents.append(s[start:i])
                i += 1
        else:
            i += 1
    return contents


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


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string

def math_acc(line):
    pred = line['text']
    target = line['additional_info']['solution']
    # if '\\text{' in pred:
    #     pred = pred.replace('\\text{', '').rstrip('}')
    pred = _remove_right_units(pred)
    target_answer = extract_bbox_content(target)[0]
    target_answer = _remove_right_units(target_answer)
    if '\\text{' in target:
        target = target.replace('\\text{', '').rstrip('}')
    
    target_answer = math_utils._strip_string(target_answer)
    pred_answer = extract_answer_content(pred)
    pred_answer = [x for x in pred_answer if x is not None]
    
    if pred_answer == []:
        return 0
    pred_answer = pred_answer[0]
    # maybe a coordinate
    if "," in pred_answer and "(" in target_answer and ")" in target_answer and "(" not in pred_answer and ")" not in pred_answer:
        # print(pred_answer)
        temps = pred_answer.split(",")
        
        pred_answer = "(" + ",".join([math_utils._strip_string(x) for x in temps]) + ")"
    else:
        pred_answer = math_utils._strip_string(pred_answer)
    if "p.m" in pred_answer and not pred_answer.endswith("."):
        pred_answer = pred_answer + "."
    if "a.m" in pred_answer and not pred_answer.endswith("."):
        pred_answer = pred_answer + "."
    # if "p.m." in target_answer or "a.m." in target_answer:
    #     pred_answer = pred_answer.replace("p.m")
    line['additional_info']['pred_answer'] = pred_answer
    line['additional_info']['target_answer'] = target_answer

    # pred_answer = extract_bbox_content(pred)
    return 1 if target_answer in pred_answer else 0
    # target_answer = _remove_right_units(target_answer)




def code_acc(line):
    cwd = os.getcwd()
    text = line['text']


    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    
    # 如果找到匹配项，则提取并打印
    if match:
        extracted_content = match.group(1)
    else:
        extracted_content = text

    additional_info = line['additional_info']
    # function_name = additional_info['function_name']
    test = additional_info['test']
    executable_code = extracted_content
    if isinstance(test, str):
        test_code = executable_code + "\n" + test
    else:
        test_code = executable_code + "\n" + "\n".join(test)
    
    if "def " not in test_code:
        test_code = additional_info.get("function_name", "") + test_code 
        
    if additional_info.get("entry_point", None) is not None:
        test_code = test_code + "\n\n" + f"check({additional_info['entry_point']})"
    

    with tempfile.TemporaryDirectory() as tempdir_name:
        tempdir_name = Path(tempdir_name)
        with open(tempdir_name / "program.py", "w", encoding="UTF-8") as f:
            f.write(test_code)
        os.chdir(tempdir_name)
        

        p = Popen(f'python program.py', shell=True, stdout=PIPE, stderr=PIPE)
        time_limit = 15  # seconds
        scores = 1
        try:
            stdout, stderr = p.communicate(timeout=time_limit)
        except TimeoutExpired:
            # Linux
            # os.killpg(p.pid, signal.SIGTERM)
            # Windows
            os.system("kill {pid}".format(pid=p.pid))
            scores = 0
        else:
            if stderr:
                scores = 0
    

    os.chdir(cwd)
    return scores

def gsm8k_acc(line):
    # extract answer after #### 
    pred = line['text']
    target = line['additional_info']['answer']

    index = target.find("####")
    target_answer = target[index+4:].strip()
    

    pred_answer = extract_answer_content(pred)
    # import pdb
    # pdb.set_trace()
    # if index != -1:
    #     pred_answer = pred[index + 4:].strip()  # Extract answer after "####" and strip any leading or trailing whitespace
    # else:
    #     pred_answer = pred
    # index = target.find("####")
    # target_answer = target[index + 4:].strip()
    if pred_answer is not None:
        return 1 if any(target_answer in str(x) for x in  pred_answer) else 0
    else:
        return 0

def gsmic_acc(line):
    # extract answer after #### 
    pred = line['text']
    target = line['additional_info']['answer']


    target_answer = target
    

    pred_answer = extract_answer_content(pred)
    # import pdb
    # pdb.set_trace()
    # if index != -1:
    #     pred_answer = pred[index + 4:].strip()  # Extract answer after "####" and strip any leading or trailing whitespace
    # else:
    #     pred_answer = pred
    # index = target.find("####")
    # target_answer = target[index + 4:].strip()
    if pred_answer is not None:
        return 1 if any(target_answer in str(x) for x in pred_answer) else 0
    else:
        return 0

def sum_acc(line):
    pred = line['text']
    target = line['additional_info']['answer']
    rouge = Rouge()
    rouge_score = rouge.get_scores(pred, target)
    rouge_1 = rouge_score[0]['rouge-1']['r']
    rouge_2 = rouge_score[0]['rouge-2']['r']
    rouge_l = rouge_score[0]['rouge-l']['r']
    return rouge_1, rouge_2, rouge_l
    

def mmedbench_acc(line):
    pred = line['text']
    pred = re.findall(r'[A-E]', pred)[0]

    answer = line['additional_info']['answer_idx']

    return 1 if pred == answer else 0 

def bbh_acc(line):
    pred = line['text']
    answer = line['additional_info']['target']
    if "(" in pred and ")" in pred:
        # extract the content in () [maybe many], and select the one which is a single capital letter
        pred = re.findall(r'\((.*?)\)', pred)
        for p in pred:
            if p in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                pred = [f"({p})"]
                break
    else:
        pred = extract_answer_content(pred)

    return 1 if any(answer in str(x) for x in pred) else 0

def apps_acc(line):
    text = line['text']
    match = re.search(r"```python(.*?)```", text)

    if match:
        extracted_content = match.group(1)
    else:
        extracted_content = text
    additional_info = line['additional_info']
    input_output = additional_info['input_output']
    # try:
    #     input_output = json.loads(input_output)
    # except:
    #     return None
    # input_output = json.loads(input_output)

    inputs = input_output['inputs']
    outputs = input_output['outputs']
    test_code = extracted_content 
    assert len(inputs) == len(outputs)
    
    ff = tempfile.NamedTemporaryFile(mode='w')
    ff.write(test_code)
    name = ff.name 
    scores = 1
    for i in range(len(inputs)):
        cur_input = inputs[i]
        cur_output = outputs[i]
        
        p = Popen(f'python {name} < {cur_input}', shell=True, stdout=PIPE, stderr=PIPE)
        time_limit = 15  # seconds
        try:
            stdout, stderr = p.communicate(timeout=time_limit)
        except TimeoutExpired:
            # Linux
            # os.killpg(p.pid, signal.SIGTERM)
            # Windows
            # Popen("TASKKILL /F /PID {pid} /T".format(pid=p.pid))
            scores = 0
            break
        if stderr:
            scores = 0
            break
        if stdout.strip() != cur_output.strip():
            scores = 0
            break
    ff.close()
    return scores

def triviaqa_acc(line):
    pred = line['text']
    answers = line['additional_info']['answer']
    for answer in answers:
        if pred == answer:
            return 1 
    return 0


def mc_acc(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if pred.endswith("."):
        pred = pred[:-1]
    return 1 if pred == answer else 0

winogrande = mmlu_acc = arc_acc = cmmlu_acc = ceval_acc = mc_acc

def commonsense_qa_acc(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if len(pred) == 1:
        return 1 if pred == answer else 0
    extract_pred = extract_answer_content(pred)
    if extract_pred is not None:
        return 1 if any(answer in str(x) for x in extract_pred) else 0
    else:
        return 0

    
def logiqa_en(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if len(pred) == 1:
        return 1 if pred == answer else 0
    extract_pred = extract_answer_content(pred)
    if extract_pred is not None:
        if "(" in extract_pred:
            extract_pred = extract_pred[extract_pred.index("(") + 1: extract_pred.index(")")]
    else:
        return 0
    return 1 if any(answer in str(x) for x in extract_pred) else 0

def logiqa_zh(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if len(pred) == 1:
        return 1 if pred == answer else 0
    match = re.search(r'答案是 (.*?)\。', pred, re.IGNORECASE)

    extract_pred = match.group(1) if match else None
    # extract_pred = extract_answer_content(pred)
    if extract_pred is not None:
        if "(" in extract_pred:
            extract_pred = extract_pred[extract_pred.index("(") + 1: extract_pred.index(")")]
    return 1 if answer == extract_pred else 0

def svamp_acc(line):
    pred = line['text']
    target = line['additional_info']['answer']

    target_answer = target
    pred_answer = extract_bbox_content(pred)

    # print(target)
    # print(target_answer)
    # print(pred)

    if pred_answer != []:
        pred_answer = pred_answer[0]
        if isinstance(target_answer, float):
            pred_answer = normalize(pred_answer) #if pred_answer is not None else float("-inf")

        if isinstance(target_answer, str) and isinstance(pred_answer, str): # target type = str
            return 1.0 if target_answer in pred_answer else 0.0
        
        elif isinstance(pred_answer, str): # target type = float
            return 1.0 if pred_answer in str(target) else 0.0
        
        elif isinstance(pred_answer, float):
            if abs(pred_answer - target_answer) < 1e-3:
                return 1.0
            else:
                return 0.0

        return 0
    else:
        if "the answer is" in pred or "The answer is":
            pred_answer = extract_answer_content(pred)
            try:
                return 1.0 if any(abs(target_answer - float(x)) < 1e-3 for x in pred_answer) else 0.0
            except:
                if abs(target_answer - int(target_answer)) < 1e-4:
                    # the answer can be expressed as int 
                    target_answer = str(int(target_answer))
                    return 1.0 if any(target_answer in str(x) for x in pred_answer) else 0.0
                else:
                    # the answer is float and the pred_answer cannot be converted to float 
                    return 0
        else:
            pred_answer = pred[len(pred) // 2:]
            return 1.0 if target_answer in pred_answer else 0.0

def multiplechoice_acc(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if len(pred) == 1:
        return 1 if pred == answer else 0
    extract_pred = extract_answer_content(pred)
    if extract_pred is not None:
        return 1 if any(answer in str(x) for x in extract_pred) else 0
    else:
        return 0


def multiplechoice_zh_acc(line):
    pred = re.search(r'答案为(.*?)$', line['text'].rsplit("\n\n\n", 1)[0])
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        # import pdb
        # pdb.set_trace()
        answer = line['additional_info']['answer']
        if "错，为本题正确答案" in line['text'] and f"{answer}错，为本题正确答案" in line['text']:
            return 1
        else:
            all_index = "ABCDE"
            for answer in line['additional_info']['answer']:
                all_index = all_index.replace(answer, "")
                if f"{answer}对" not in line['text']:
                    return 0
            # if len(line['additional_info']['answer']) > 1:
            if True:
                for o_answer in all_index:
                    if f"{o_answer}对" in line['text']:
                        return 0
            return 1

    else:
        pred = pred[0]
    
    all_index = "ABCDE"
    answer_list = line['additional_info']['answer']
    for answer in answer_list:
        all_index = all_index.replace(answer, "")
        if answer not in pred:
            return 0
    # if len(answer_list) > 1:
    if True:
        for o_answer in all_index:
            if f"{o_answer}" in pred:
                return 0
    return 1




def mmedbench_en_cot_acc(line, prefix='the answer is'):
    text = line['text']
    if not text.endswith("."):
        text = text + "."
    pred = re.search(prefix + r' (.*?)\.', text, re.IGNORECASE)
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer_idx']

    return 1 if f"{answer}." in pred else 0 

def mmedbench_zh_cot_acc(line):
    text = line['text']
    if not text.endswith("."):
        text = text + "."
    pred = extract_answer_content(text, "The answer is")
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer_idx']

    return 1 if answer in pred else 0 

def numglue_acc(line):
    pred = line['text']
    if not pred.endswith("."):
        pred = pred + "."
    target_answer = line['additional_info']['answer']

    pred_answer = extract_answer_content(pred)
    
    if pred_answer is not None:
        # try:
        #     target_answer = float(target_answer)
        #     pred_answer = [float(x) for x in pred_answer]
        #     return 1 if any(abs(target_answer - x) < 1e-3 for x in pred_answer) else 0 
        # except:
        # it is a string answer
        pred_answer = [x.replace(",", "") for x in pred_answer if x is not None]
        target_answer = str(target_answer)
        return 1 if any(target_answer in str(x) for x in pred_answer) else 0
        
    else:
        return 0

def tydiqa_acc(line):
    pred = line['text']
    if not pred.endswith("."):
        pred = pred + "."
    ground_truths = line['additional_info']['answers']
    def f1_score(prediction, ground_truth):
        prediction_tokens = general_postprocess(prediction).split()
        ground_truth_tokens = general_postprocess(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    def em_score(prediction, ground_truth):
        return (general_postprocess(prediction) == general_postprocess(
            ground_truth))
    if "The answer is" not in pred:
        pred_answer = pred 
    else:
        pred_answer = extract_answer_content(pred)
        pred_answer = pred_answer[0]
    if pred_answer is None:
        return 0
    scores = []
    for ground_truth in ground_truths:
        score = f1_score(pred_answer, ground_truth)
        scores.append(score)
    
    score = max(scores)
    return score


def information_extraction_acc(line):
    pred = line['text']
    if not pred.endswith("."):
        pred = pred + "."
    target_answer = line['additional_info']['answer']
    all_answers = line['all_answer']
    if "The answer is " not in pred:
        pred_answer = [pred]
    else:
        pred_answer = extract_answer_content(pred)
    if pred_answer is not None:

        pred_answer = pred_answer[0]
        if pred_answer is None:
            return 0
        if isinstance(target_answer, str):
            return target_answer in pred_answer
        else:
            for each in target_answer:
                if each in pred_answer:
                    return 1
            return 0

    else:
        return 0

def medsins_acc(line):
    pred = line['text']
    if not pred.endswith("."):
        pred = pred + "."
    target_answer = line['additional_info']['answer']
    if "The answer is " not in pred:
        pred_answer = [pred]
    else:
        pred_answer = extract_answer_content(pred)
    if pred_answer is not None:

        pred_answer = pred_answer[0]
        if pred_answer is None:
            return 0
        if len(target_answer) == 1:
            clean_pred = pred_answer.replace("'", "").replace('"', '')
            return int(target_answer in clean_pred)
        else:
            clean_answer = target_answer.lower()
            clean_pred = pred_answer.lower()
            return int(clean_answer in clean_pred)
    else:
        return 0
        # if isinstance(target_answer, str):
        #     return target_answer in pred_answer
        # else:
        #     for each in target_answer:
        #         if each in pred_answer:
        #             return 1
        #     return 0
    
METRIC_FUNC_MAPPING = {


    "mmedbench_en": mmedbench_acc,
    "mmedbench_zh": mmedbench_acc,





    "cmmlu": cmmlu_acc,






    "mmedbench_en_cot": mmedbench_en_cot_acc,







    "mmedbench_zh_cot": mmedbench_zh_cot_acc,


    "MedMCQA": multiplechoice_acc,
    "MedQA": multiplechoice_acc,

    'med_mmlu': multiplechoice_acc,
    'pubmedqa': multiplechoice_acc,
    "bioasq": multiplechoice_acc,
    "pubhealth": multiplechoice_acc,
    "biomrc": multiplechoice_acc,

    "healthfact": medsins_acc,
    "drugdose": medsins_acc,
    "ddxplus": medsins_acc,
    "seer": medsins_acc,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=False)
    parser.add_argument("--test_num", type=int, default=-1)
    args = parser.parse_args()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]
    if args.test_num != -1:
        questions = questions[:args.test_num]
    total_num = len(questions)
    total_score = 0
    rouge_score = [[], [], []]

    dataset_name = ""
    i = 1
    while dataset_name not in METRIC_FUNC_MAPPING:
        dataset_name = args.input_file.split("/")[-i].replace(".jsonl", "")
        i += 1
    acc_func = METRIC_FUNC_MAPPING[dataset_name]
    wrong_idx = []
    for line in tqdm(questions, total=total_num):
        scores = acc_func(line)
        if isinstance(scores, tuple):
            rouge_1, rouge_2, rouge_l = scores
            rouge_score[0].append(rouge_1)
            rouge_score[1].append(rouge_2)
            rouge_score[2].append(rouge_l)
            continue
        if scores is None:
            total_num -= 1
            wrong_idx.append(line)
            continue
        total_score += scores
        if scores == 0:
            wrong_idx.append(line)
    if rouge_score[0]:
        print(f"Rouge-1: {sum(rouge_score[0]) / len(rouge_score[0])}")
        print(f"Rouge-2: {sum(rouge_score[1]) / len(rouge_score[1])}")
        print(f"Rouge-l: {sum(rouge_score[2]) / len(rouge_score[2])}")
        exit(0)
    avg_acc = total_score / total_num
    print(f"Acc in {dataset_name}: {avg_acc}")
    
    # save the result to a json file 
    json_file = args.input_file.replace(".jsonl", "_result.json")
    infos = {
        "acc": avg_acc,
        "pass@1": avg_acc,
    }
    
    # if args.output_file:
    #     client.write_json(wrong_idx, args.output_file, ensure_ascii=False, indent=4)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(wrong_idx, f, ensure_ascii=False, indent=2)