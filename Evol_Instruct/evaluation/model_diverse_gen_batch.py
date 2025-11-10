import argparse

import os
import json
from tqdm import trange

# import shortuuid

from Evol_Instruct.solver.cot_solver import CoTSolver

from Evol_Instruct.solver.sc_solver import SCSolver

from Evol_Instruct.models.builder import (
    load_dpo_model,
    load_pretrained_model,
    load_value_model,
)

from Evol_Instruct.utils.utils import (
    disable_torch_init,
    get_model_name_from_path,
    get_chunk,
)
from Evol_Instruct.evaluation.generate_utils import (
    task_specific_prompt_mapping,
    CustomDataset,
    set_tokenizer,
)

from torch.utils.data import DataLoader
import pandas as pd
from Evol_Instruct import client
from copy import deepcopy


# DataLoader
def create_data_loader(questions, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return data_loader


def convert_to_json(questions):

    questions = questions.to_dict(orient="records")
    return questions


four_choices_datasets = ["med_mmlu", "MedMCQA"]
five_choices_datasets = ["MedQA_cot"]
three_choices_datasets = ["pubmedqa", "pubhealth"]
two_choices_datasets = ["bioasq"]


class Generation:
    def __init__(self, args):
        self.args = args
        self.questions, self.ans_file = self.initialize(args)
        self.tokenizer, self.server = self.obtain_models(args)
        model_path = os.path.expanduser(args.model_path)
        if "32b" in model_path.lower() or (
            args.model_base is not None and "32b" in args.model_base.lower()
        ):
            args.batch_size = 4
        if "truthfulqa_mc1" in self.dataset_name:
            args.batch_size = 1
        if "biomrc" in self.dataset_name:
            args.batch_size = 1
        task_specific_prompt = task_specific_prompt_mapping.get(self.dataset_name, "")

        self.dataset = CustomDataset(
            self.questions,
            batch_size=args.batch_size,
            task_specific_prompt=task_specific_prompt,
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            add_few_shot=args.add_few_shot,
            num_samples=args.fewshot_samples,
        )

        if self.dataset_name in four_choices_datasets:
            choices_word = ["A", "B", "C", "D"]
        elif self.dataset_name in five_choices_datasets:
            choices_word = ["A", "B", "C", "D", "E"]
        elif self.dataset_name in three_choices_datasets:

            choices_word = ["A", "B", "C"]
        elif self.dataset_name in two_choices_datasets:
            choices_word = ["A", "B"]
        elif self.dataset_name in ["biomrc"]:
            choices_word = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        else:

            choices_word = None

        self.sampling_params = {
            "temperature": args.temperature,
            "n": args.sampling_numbers,
            "top_p": args.top_p,
            "max_tokens": args.max_new_tokens,
        }
        if self.args.model_base is None or self.args.model_base == "None":
            infer_answer_max_tokens = 1
        else:
            infer_answer_max_tokens = None

        if self.args.sampling_strategy in ["greedy"]:
            self.solver = CoTSolver(
                self.server,
                self.ans_file,
                choices_word=choices_word,
                cot_prompt=self.dataset.cot_prompt,
                infer_answer_max_tokens=infer_answer_max_tokens,
            )
        elif self.args.sampling_strategy in ["sc"]:
            self.solver = SCSolver(
                self.server,
                self.ans_file,
                choices_word=choices_word,
                cot_prompt=self.dataset.cot_prompt,
                infer_answer_max_tokens=infer_answer_max_tokens,
            )

    def infer(self):
        args = self.args
        args_file = os.path.join(os.path.dirname(self.args.answers_file), "args.json")
        with open(args_file, "w") as f:
            json.dump(vars(args), f, indent=4)

        for idx in trange(len(self.dataset)):
            items = self.dataset[idx]
            new_items = self.solver.generate_response(
                items,
                value_model_type=args.value_model_type,
                expand_way=args.tot_expand_way,
                **self.sampling_params,
            )
            self.solver.save_response(new_items)

        self.ans_file.close()

    def initialize(self, args):
        dataset_name = args.question_file.split("/")[-1].split(".")[0]

        print(args)

        if args.question_file.endswith(".csv"):
            questions = pd.read_csv(args.question_file)
            questions = convert_to_json(questions)
        elif args.question_file.endswith(".jsonl"):
            questions = client.read_jsonl(os.path.expanduser(args.question_file))
        else:
            # a json file
            questions = client.read_json(os.path.expanduser(args.question_file))
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        answers_file = os.path.expanduser(args.answers_file)
        if os.path.dirname(answers_file) != "":
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        if args.resume and os.path.exists(answers_file):
            data = client.read_jsonl(answers_file)
            current_file_id = set([item["id"] for item in data])
            questions = [q for q in questions if q["id"] not in current_file_id]

            ans_file = open(answers_file, "a", encoding="utf-8")
        else:
            ans_file = open(answers_file, "w", encoding="utf-8")
        if len(questions) == 0:
            exit(0)
        self.dataset_name = dataset_name
        return questions, ans_file

    def obtain_models(self, args):
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)

        args.gpu_memory_usage = 0.9

        server = load_pretrained_model(
            model_path, args.model_base, model_name, args=args
        )
        tokenizer = server.tokenizer
        tokenizer = set_tokenizer(tokenizer)
        # print(tokenizer)
        self.model_name = model_name

        return tokenizer, server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)


    parser.add_argument("--add_few_shot", action="store_true")
    parser.add_argument("--fewshot_samples", type=int, default=1)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")

    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-logit-bias", action="store_true")
    parser.add_argument("--logit-score", default=100.0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument(
        "--sampling_strategy", type=str, default="greedy", choices=["greedy", "sc"]
    )

    parser.add_argument("--sampling_numbers", type=int, default=1)

    parser.add_argument(
        "--value_model_type", type=str, default="orm", choices=["orm", "prm"]
    )

    parser.add_argument("--infer-answer", action="store_true")

    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    generator = Generation(args)
    generator.infer()
    # eval_model(args)
