from collections import defaultdict
from typing import Counter

import scipy
import scipy.stats
from Evol_Instruct import client
import numpy as np
import argparse 
from Evol_Instruct.evaluation.eval_em import METRIC_FUNC_MAPPING



def compute_dataset_with_method(data_path, method, n=-1):
    data = client.read(data_path)
    dataset_name = ""

    acc = 0
    for line in data:
        all_answer = line['all_answer']
        
        
        all_answer = [(answer, score) for answer, score in all_answer if answer]
        if n != -1:
            all_answer = all_answer[:n]
        if method == 'vote-sc':
            text_answers = [answer[0] for answer in all_answer]
            final_answer = Counter(text_answers).most_common(1)[0][0]
        elif method == 'max':
            final_answer = max(all_answer, key=lambda x: x[1])[0]
        elif method == 'vote-sum':
            # text_answers = 
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += score if isinstance(score, float) else score[-1]
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'vote-mean':
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += score if isinstance(score, float) else score[-1]
            answer_count = Counter([answer for answer, _ in all_answer])
            for answer in answer_dict:
                answer_dict[answer] /= answer_count[answer]
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'sc':
            if isinstance(all_answer[0], list):
                all_answer = [answer[0] for answer in all_answer]
            final_answer = Counter(all_answer).most_common(1)[0][0]
        elif method == 'prm-mean-vote-sum':
            answer_dict = defaultdict(float)
            # all_answer = 
            for answer, score in all_answer:
                answer_dict[answer] += np.mean(score[1:])
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'prm-prod-vote-sum':
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += np.prod(score[1:])
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'prm-gmean-vote-sum':
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += scipy.stats.gmean(score[1:])
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'prm-gmean-vote-mean':
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += scipy.stats.gmean(score[1:])
            answer_count = Counter([answer for answer, _ in all_answer])
            for answer in answer_dict:
                answer_dict[answer] /= answer_count[answer]
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'prm-min-vote-sum':
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += np.min(score[1:])
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'prm-sum-vote-sum':
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += np.sum(score[1:])
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'prm-sum-vote-mean':
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += np.sum(score[1:])
            answer_count = Counter([answer for answer, _ in all_answer])
            for answer in answer_dict:
                answer_dict[answer] /= answer_count[answer]
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'prm-vote-sum':
            answer_dict = defaultdict(float)
            for answer, score in all_answer:
                answer_dict[answer] += score[-1]
            final_answer = max(answer_dict, key=answer_dict.get)
        elif method == 'prm-meanmax':
            final_answer = max(all_answer, key=lambda x: np.mean(x[1][1:]))[0]
        elif method == 'prm-minmax':
            final_answer = max(all_answer, key=lambda x: np.min(x[1][1:]))[0]
        elif method == 'prm-prodmax':
            final_answer = max(all_answer, key=lambda x: np.prod(x[1][1:]))[0]
        elif method == 'prm-max':
            final_answer = max(all_answer, key=lambda x: x[1][-1])[0]
        elif method == 'prm-gmeanmax':
            final_answer = max(all_answer, key=lambda x: scipy.stats.gmean(x[1][1:]))[0]
        ground_truth = line['additional_info']['answer']
        
        if ground_truth.lower() in final_answer.lower():
            acc += 1
    return acc / len(data)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument('--n', type=int, default=-1)
    args = parser.parse_args()
    acc = compute_dataset_with_method(args.data_path, args.method, args.n)
    acc_in_percent = acc * 100
    print(acc_in_percent)
