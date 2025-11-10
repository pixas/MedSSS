from copy import deepcopy
import json
from typing import Any

from openai import chat

from Evol_Instruct.MCTS.tree_node import MedMCTSNode

from Evol_Instruct.models.vllm_support import VLLMServer, chat_prompt
from Evol_Instruct.prompts.prompt_template import mcts_prompts
from Evol_Instruct.prompts.examples import *
from Evol_Instruct.actions.actions_register import register

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList 
from Evol_Instruct.utils.utils import LogitBiasProcess
import re

DEF_INNER_ACT_OBS = "OK"
INNER_ACT_KEY = "response"

class BaseAction:
    def __init__(self, action_name: str, action_desc: str):
        self.action_name = action_name 
        self.description = action_desc
    
    def __repr__(self):
        string = json.dumps(self.__dict__)
        return string
    
    def call_prompt(self, node: MedMCTSNode):
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Any:
        raise NotImplementedError


@register("Reason", "base")
class Reason(BaseAction):
    def __init__(self) -> None:
        action_name = "Reason"
        action_desc = "This action is conducted if the task requires further one-step reasoning or no previous reasoning steps available. "

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
        )
        
        self.details = "Think critically about the problem and answer with concise, accurate reasoning. Please ensure your reasoning is thorough and elaborate, breaking down each step of your thought process.\n"
    
    def call_prompt(self, node: MedMCTSNode, few_shot=0) -> str:
        prompt = self.details
        if few_shot > 0:
            few_shot_prompt = "Reasoning Example:\n" + "\n\n".join(mcts_example[:few_shot]) + "\n\n"
        else:
            few_shot_prompt = ""
        prompt = few_shot_prompt + mcts_prompts['prefix'] + prompt + f"Problem: {node.problem}"

        return prompt

    def __call__(self, node: MedMCTSNode, server: VLLMServer, few_shot=0, first=True, pre_gen_texts=None, **kwargs):
        step_count = len(node.trace)
        if first:
            cur_prompt = self.call_prompt(node, few_shot=few_shot)
            cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + node.obtain_reasoning_steps()[0] + f"\n\nStep {step_count}:{'' if pre_gen_texts is None else pre_gen_texts}"
        else:
            cur_prompt = node.problem
            cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + node.obtain_reasoning_steps()[0] + f"\n\nStep {step_count}:{'' if pre_gen_texts is None else pre_gen_texts}"
        # cur_prompt = self.call_prompt(node, few_shot=few_shot)
        # cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + f"<steps>\nStep {step_count}:"
        reasoning_steps = server([cur_prompt], wrap_chat=False, **kwargs)
        result = [x.strip().rstrip(".") + "." for x in reasoning_steps[0]]
        
        return result


@register("Finish", "base")
class Finish(BaseAction):
    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = """This action is conducted if all reasoning steps are complete enough and a final answer can be provided. """

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
        )

        self.details = """Use thorough and elaborate steps to complete your reasoning. Conclude the task by stating: "The answer is {{answer}}"."""

    def call_prompt(self, node: MedMCTSNode, few_shot=0) -> str:
        prompt = self.details
        if few_shot > 0:
            # few_shot_prompt = "<example>\n" + "\n\n".join(mcts_example[:few_shot]) + "\n</example>\n\n"
            few_shot_prompt = "Reasoning Example:\n" + "\n\n".join(mcts_example[:few_shot]) + "\n\n"
        else:
            few_shot_prompt = ""
        # prompt = few_shot_prompt + mcts_prompts['prefix'] + prompt + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" 
        prompt = few_shot_prompt + mcts_prompts['prefix'] + prompt + f"Problem: {node.problem}"
        return prompt

    def __call__(self, node: MedMCTSNode, server: VLLMServer, few_shot=0, first=True, direct_output=False, **kwargs):
        step_count = len(node.trace)
        if first:
            # cur_prompt = self.call_prompt(node, few_shot=few_shot)
            # cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + f"<finish>\nStep {step_count}:"
            # kwargs['stop'] = ['</finish>']
            cur_prompt = self.call_prompt(node, few_shot=few_shot)
            cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + node.obtain_reasoning_steps()[0] + f"\n\nStep {step_count}:"
        else:
            cur_prompt = "Derive the answer of the problem and concude the task by stating: \"The answer is {{answer}}\".\n" + node.problem
            cur_prompt = node.problem
            cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + node.obtain_reasoning_steps()[0] + f"\n\nStep {step_count}:"
        if direct_output:
            kwargs['stop'] = [server.tokenizer.eos_token]
        reasoning_steps = server([cur_prompt], wrap_chat=False, **kwargs)
        # new_steps = []
        # for each_step in reasoning_steps[0]:
        #     answer = extract_template(each_step, "answer")
        #     if answer is None:
        #         infer_answer(server.tokenizer, server.model,
        #                      [cur_prompt], [each_step], server.lora_request,
        #                      )
        #     else:
        #         new_steps.append(each_step)
        # if "The answer is" not in 
        # reasonig_steps = [step for step in reasoning_steps[0]]
        result = [x.strip().rstrip(".") + "." for x in reasoning_steps[0]]
        return result

@register("Reflect", "base")
class Reflect(BaseAction):
    def __init__(self) -> None:
        action_name = "Reflect"
        action_desc = """This action is conducted if the previous reasoning steps contain ambiguity or mistakes. Only choose "Reflect" if there are prior reasoning steps."""

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
        )
        self.details = """Review the previous reasoning step critically to identify any necessary corrections, refinements, or enhancements. Generate **one single reasoning step** to improve the coherence, accuracy, or depth of the previous reasoning. Ensure this revision aligns with the problem's intent and brings clarity or correctness to the analysis. This step should hold the same format as previous reasoning steps."""

    def call_prompt(self, node: MedMCTSNode, few_shot=0) -> str:
        prompt = self.details
        if few_shot > 0:
            few_shot_prompt = "<example>\n" + "\n\n".join(mcts_refine_example[:few_shot]) + "\n</example>\n\n"
        else:
            few_shot_prompt = ""
        prompt = few_shot_prompt + mcts_prompts['prefix'] + prompt + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" 
        return prompt

    def __call__(self, node: MedMCTSNode, server: VLLMServer, few_shot=0, first=True, **kwargs):
        step_count = len(node.trace)
        # cur_prompt = self.call_prompt(node, few_shot=few_shot)
        # cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + f"<reflect>\nStep {step_count}: Reflect: "
        # reasoning_steps = server([cur_prompt], wrap_chat=False, stop=["</reflect>", "\n\n"] + [server.tokenizer.eos_token], **kwargs)
        # incorrect_answer = extract_template(node.obtain_reasoning_steps()[0], "answer")
        if False:
            cur_prompt = self.call_prompt(node, few_shot=few_shot)
            cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + f"<reflect>\nStep {step_count}:"
            kwargs['stop'] = ['</reflect>']
        else:
            trigger_word = "But wait, "
            cur_prompt = f"The answer derived by the reasoning steps is wrong for the problem, which suggests that the cut-off reasoning steps are somewhat of mistakes or irrelevant information for solving the given problm. Review the previous reasoning steps critically to identify any necessary corrections, refinements, or enhancements. Generate revision steps to mprove the coherence, accuracy, or depth of the previous reasoning. Start your reasoning with 'Step {step_count}: {trigger_word}'.\nProblem: {node.problem}\nWrong steps: {node.obtain_reasoning_steps()[0]}"
            cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + trigger_word
        # post processing, only keep the first reasoning step
        reasoning_steps = server([cur_prompt], wrap_chat=False, **kwargs)
        reasoning_steps = [x.split("\n\n<problem>")[0].split("\n\n<steps>")[0].split("\n\nStep")[0].split("\n\n<step>")[0] \
            for x in reasoning_steps[0]]
        reasoning_steps = [trigger_word + x for x in reasoning_steps] 
        return reasoning_steps

@register("Think", "plan")
class Think(BaseAction):
    def __init__(self, available_actions: dict[str,BaseAction]) -> None:
        action_name = "Think"
        action_desc = "Given all previous reasoning steps, decide on the most suitable action to take next for solving the given problem"
        
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,

        )
        self.available_actions = list(available_actions.values())
        # self.actions = [eval(f"{action}()") for action in self.available_actions]
        self.action_description = [act.description for act in self.available_actions]
        self.available_action_names = [act.action_name for act in self.available_actions]
        self.details = f"""Output your choice using the format: "The action is {{action}}." {{action}} can only takes in {self.available_action_names}. Do not output other information."""
    
    def call_prompt(self, node: MedMCTSNode) -> str:
        action_docs = [f"{i + 1}. {act.action_name}: {act.description}" for i, act in enumerate(self.available_actions)]
        action_docs = "\n".join(action_docs)
        prompt = self.description + "\n\n" + action_docs + "\n" + self.details
        
        prompt = mcts_prompts['prefix'] + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" + prompt
        return prompt
        

    def __call__(self, node: MedMCTSNode, server: VLLMServer, **kwargs):
        node_prompt = self.call_prompt(node)
        cur_prompt = chat_prompt([node_prompt], server.tokenizer)[0] + f"The action is "
        logits_processors = LogitsProcessorList()
        action_index = [server.tokenizer.encode(x)[0] for x in self.available_action_names]
        logits_processors.append(LogitBiasProcess(action_index))
        cur_args = deepcopy(kwargs)
        cur_args['max_tokens'] = 1
        steps = server([cur_prompt], system='You are a helpful assistant.', logits_processors=logits_processors, **cur_args)
        next_actions = steps[0]
        # output = server(prompt, **kwargs)
        return next_actions
    
@register("Refine", "refine")
class Refine(Reason):
    def __init__(self):
        action_name = "Refine"
        action_desc = "This action is conducted if the task requires to refine the previous reasoning steps. "

        super().__init__(
            # action_name=action_name,
            # action_desc=action_desc,
        )
        self.action_name = action_name
        # self.details = "Continual the reasoning steps of given medical problems and provided error hints"

    def call_prompt(self, node: MedMCTSNode, few_shot=0) -> str:
        prompt = self.details 
        if few_shot > 0:
            few_shot_prompt = "<example>\n" + "\n\n".join(mcts_refine_example[:few_shot]) + "\n</example>\n\n"
        else:
            few_shot_prompt = ""
        prompt = few_shot_prompt + mcts_prompts['prefix'] + prompt + f"<problem>\n{node.problem}\n</problem>\n\n<steps>\n{node.obtain_reasoning_steps()[0]}</steps>\n\n"
        # cur_prompt = chat_prompt([prompt], tokenizer=)
        return prompt

    def __call__(self, node: MedMCTSNode, server: VLLMServer, few_shot=0, **kwargs):
        previous_reasoning_steps, step_count = node.obtain_reasoning_steps()
        cur_prompt = self.call_prompt(node, few_shot=few_shot)
        
        cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + f"<steps>\nThere are mistakes in the reasoning steps. Let's rethink this."
        reasoning_steps = server([cur_prompt], wrap_chat=False, stop=["</steps>", server.tokenizer.eos_token], **kwargs)
        reasoning_steps = [[step.strip(" ") for step in reasoning_step] for reasoning_step in reasoning_steps]
        return reasoning_steps[0]
    
    
class Simulation(BaseAction):
    def __init__(self) -> None:
        action_name = "Simulation"
        action_desc = "This action is conducted if the task requires a simulation of a specific scenario or process. "
        params_doc = {
            INNER_ACT_KEY: """Given all previous reasoning steps, generate the final answer. Conclude the task by stating: "The answer is {{answer}}". Ensure that this answer is consistent with the prior steps and fully addresses the task question."""
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
        )

    def call_prompt(self, node: MedMCTSNode) -> str:
        prompt = self.params_doc[INNER_ACT_KEY]
        prompt = mcts_prompts['prefix'] + prompt + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" 
        return prompt

    def __call__(self, node: MedMCTSNode, server: VLLMServer, **kwargs):
        prompt = self.call_prompt(node)
        output = server(prompt, **kwargs)
        return output

@register("Decompose", 'other')
class Decompose(BaseAction):
    def __init__(self) -> None:
        action_name = "Decompose"
        action_desc = "This action is conducted if the task requires to break down the problem into smaller, more manageable parts. "

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
        )
        # self.details = "Break down the problem into smaller, more manageable parts. Provide a clear and concise decomposition of the problem into smaller components. Output the decomposition in ordered steps, each focusing on a specific aspect of the problem. You will be provided with previous solved steps to help you decompose the problem effectively and avoid duplication. You are required to output {question_num} sub-goals to solve the problem.\n"
        self.details = "Propose {question_num} different plans for solving the given exam question. Each subgoal should only consider the given question information, and do not require conducting more tests or medical studies. Each plan should be all complete to answer the question. Meanwhile, try your best to make each plan different with each other (e.g., different number of sub-goals, different sub-goals, etc.). Start each plan with 'Plan 1,2,3: 1.'\n"

    def call_prompt(self, node: MedMCTSNode, few_shot=0, question_num=0) -> str:
        prompt = self.details.format(question_num=question_num)
        if few_shot > 0:
            few_shot_prompt = "<example>\n" + "\n\n".join(mcts_plan_examples[:few_shot]) + "\n</example>\n\n"
            # few_shot_prompt = "Reasoning Example:\n" + "\n\n".join(mcts_example[:few_shot]) + "\n\n"
        else:
            few_shot_prompt = ""
        # prompt = few_shot_prompt + mcts_prompts['prefix'] + prompt + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" 
        prompt = few_shot_prompt + mcts_prompts['prefix'] + prompt + f"Problem: {node.problem}"
        return prompt

    def contains_step(self, sentence):
        pattern = r'Step \d+:'
        return bool(re.search(pattern, sentence))

    def find_step_index(self, sentence):
        pattern = r'Plan \d+:'
        match = re.search(pattern, sentence)
        if match:
            return match.start()  # 返回匹配项的起始索引
        return -1  # 如果没有匹配，返回 -1
    
    def __call__(self, node: MedMCTSNode, server: VLLMServer, few_shot=0, first=True, question_num=0, **kwargs):
        step_count = len(node.trace)
        if first:
            cur_prompt = self.call_prompt(node, few_shot=few_shot, question_num=question_num)
            # cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + f"<decompose>\nStep {step_count}:"
            cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0]
            kwargs['stop'] = ['</decompose>']
            # cur_prompt = self.call_prompt(node, few_shot=few_shot)
            # cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + node.obtain_reasoning_steps()[0] + f"\n\nStep {step_count}:"
        else:
            # cur_prompt = "Derive the answer of the problem and concude the task by stating: \"The answer is {{answer}}\".\n" + node.problem
            cur_prompt = self.details.format(question_num=question_num) + "\nProblem: " + node.problem
            cur_prompt = chat_prompt([cur_prompt], tokenizer=server.tokenizer)[0] + node.obtain_reasoning_steps()[0] + f"\n\nStep {step_count}:"
        subgoals = server([cur_prompt], wrap_chat=False, **kwargs)[0][0]
        
        each_line = subgoals.split("\n\n")
        output_goals = []
        add_line = False
        for line in each_line:
            # examine whether the line startswith "1." or "2."
            if (start_index:=self.find_step_index(line)) != -1:
                line = line[start_index + 6:]
                for j in range(len(line)):
                    if line[j].isdigit():
                        line = line[j:]
                        break
            if line.startswith("1."):
                output_goals.append("Decompose the problem into smaller, solvable sub-problems: " + line)
            # if not line.startswith("1."):
            #     add_line = True 
            # elif add_line and line.startswith("1."):
            # # if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            #     output_goals.append("Decompose the problem into smaller, solvable sub-problems: " + line)
            #     add_line = False
        return output_goals
        # return reasoning_steps[0]

base_actions = ['Reason', 'Finish', 'Reflect']