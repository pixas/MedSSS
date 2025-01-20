from Evol_Instruct.MCTS.tree import MCTS, MCTSConfig
from Evol_Instruct.MCTS.tree_node import LSMCTSNode
from collections import Counter, defaultdict
from copy import deepcopy
import math
import time 
from typing import Any, Callable, Union, Optional

import numpy as np
import re
import torch

# from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList 
from Evol_Instruct.MCTS.utils import extract_template
# from Evol_Instruct.actions.base_action import Finish, Reflect, Think, Reason, Refine
from Evol_Instruct.evaluation.generate_utils import infer_answer, set_tokenizer
# from Evol_Instruct.models.modeling_value_llama import LlamaForValueFunction
# from Evol_Instruct.utils.utils import LogitBiasProcess, extract_answer
from Evol_Instruct.models.vllm_support import VLLMServer, chat_prompt, get_vllm_model, vllm_clean_generate
from Evol_Instruct.prompts.prompt_template import mcts_prompts, search_prompts
from Evol_Instruct.prompts.examples import mcts_ls_examples
from Evol_Instruct import client, logger
from Evol_Instruct.MCTS.tree_node import MedMCTSNode
# from Evol_Instruct.actions.actions_register import ACTIONS_REGISTRY
# from Evol_Instruct.actions.base_action import BaseAction, base_actions
# from Evol_Instruct.actions.medrag import Medrag
# from Evol_Instruct.actions.hint import Hint

import pickle
import pdb

from Evol_Instruct.MCTS.tree_register import register_tree
from Evol_Instruct.solver.cot_solver import CoTSolver

@register_tree("LSMCTS")
class LSMCTS(MCTS):
    def __init__(self,  item, model_server: VLLMServer, config, value_function=None, training=True) -> None:
        if isinstance(item, dict):
            node = LSMCTSNode(item['conversations'][0]['value'].strip("\"").strip("\n"), 
                            reasoning_step=mcts_prompts['ls_break_down'], 
                            index=0, 
                            ground_truth=item['answer_idx'] if training else "")
        elif isinstance(item, str):
            node = MedMCTSNode(item, reasoning_step=mcts_prompts['ls_break_down'], index=0,
                               ground_truth="")
        self.root = node
        # self.lora_request = lora_request
        # self.constant = constant
        self.model_server = model_server
        self.tokenizer = self.model_server.tokenizer
        # self.model = model
        # self.tokenizer = tokenizer
        self.config = config
        self.constant = getattr(self.config.expand, "constant", 2)
        self.terminate_rule = getattr(self.config.terminate, "rule", "count")
        self.value_function = value_function
        self.training = training
        self.bear_ratio = getattr(self.config.expand, 'bear_ratio', 0.9)
        self.low_gate = getattr(self.config.expand, 'low_gate', 0.3)
        self.few_shot = getattr(self.config, "few_shot", 0)
        self.debug = getattr(self.config, "debug", False)
        self.autostep = getattr(self.config.expand, "autostep", "step")
        self.whether_refine = getattr(self.config.expand, "refine", False)
        self.terminate = False
        self.solver = CoTSolver(self.model_server, "test", )
    
    def direct_infer_simulation(self, node: LSMCTSNode, simulation_times):
        prompt =  node.build_simu_prompt()
        few_shot_prompt = "\n\n".join(mcts_ls_examples[:self.few_shot])
        cur_prompt = "Example: " + few_shot_prompt + "\n\n" + prompt
        simulation_times = max(simulation_times // (node.depth + 1), 5)
        cur_prompt = chat_prompt(prompt, self.model_server.tokenizer)[0]
        prompt = cur_prompt + node.obtain_reasoning_steps()[0]
        answers = self.model_server([prompt], n=simulation_times)
        # answers = vllm_clean_generate(self.model, prompts=[prompt], 
        #                               system='You are a helpful assistant.', 
        #                               n=simulation_times,
        #                               lora_request=self.lora_request)
        text = answers[0]
        predict_idx = [extract_template(x, 'answer') for x in text]
        is_correct = [x[0] in node.ground_truth if x is not None else False for x in predict_idx ]
        node.simulations = [int(x) for x in is_correct]
        accuracy = sum(is_correct) / len(is_correct)
        return accuracy, accuracy
    
    def run_one_iter(self, **sampling_params):
        node = self.root 
        while node.children:
            # select the child node with 
            
            child_node = self.select_child(node, self.constant)
            if child_node is None:
                # pdb.set_trace()
                # logger.debug(node.children)
                return -1
            node = child_node
        
        # logger.debug(f"Selected Node: {node}")
        if node.is_completed and node.value > 0:
            # has been rollout before
            return
        if node.visits == 0 and node.depth == 0:
            # root
            # logger.debug("Choose to expand due to root node")
            self.expand_node(node, 
                            max_children=self.config.expand.max_children,
                            bear_ratio=self.bear_ratio,
                            low_gate=self.low_gate, **sampling_params)
        elif node.visits == 0 and node.depth > 0:
            if self.value_function is not None:
                # inference mode
                value, simulation_score = self.rollout(node, self.value_func_simulation)
            else:
                # pdb.set_trace()
                value, simulation_score = self.rollout(node, self.direct_infer_simulation, simulation=self.config.simulation)
            # logger.debug(f"Choose to rollout due to leaf node not visited, Value: {value}")
            self.back_propagate(node,value, simulation_score)
        elif node.visits > 0 and node.depth > 0:
            # logger.debug("Choose to expand due to leaf node visited")
            # pdb.set_trace()
            self.expand_node(node,
                            #  lora_request=self.lora_request,
                             max_children=self.config.expand.max_children,
                             bear_ratio=self.bear_ratio,
                             low_gate=self.low_gate,  **sampling_params)
        # else:
        return 0
    
    def back_propagate(self, node: LSMCTSNode, value, simulation_score=None):
        
        while node:
            trigger = "The answer is " 
            reasoning_history = node.obtain_reasoning_steps()[0]
            if trigger in reasoning_history or trigger.lower() in reasoning_history:
                # this is the finish node, and this trajectory is never explored
                # set its father nodes to be completed if its father has only one child
                if simulation_score is not None:
                    node.correct = simulation_score >= self.bear_ratio
                else:
                    node.correct = value >= self.bear_ratio
                if not self.whether_refine:
                    node.is_completed = True
                else:
                    # node.is_completed = True
                    if node.correct:
                        # a correct node can be marked as completed
                        node.is_completed = True 
                    else:
                        # wrong node, two types
                        
                        if reasoning_history.count("The answer is") < self.config.expand.answer_refine_try:
                            node.is_completed = False 
                        else:
                            node.is_completed = True
                    # node.incorrect = simulation_score <= self.low_gate
                # node.correct = simulation_score >= self.bear_ratio if simulation_score is not None else False
            else:
                if node.children:
                    
                    node.is_completed = all(child.is_completed for child in node.children)
            # if node.parent:
            #     node.parent.is_completed = all(child.is_completed for child in node.parent.children)
            node.visits += 1
            if not node.children:
                node.value = value 
            else:
                if getattr(self.config, 'update_rule', 'default') == 'default':
                    child_mean_value = sum(child.value * child.visits for child in node.children) / sum(child.visits for child in node.children)
                    node.value = child_mean_value
                elif getattr(self.config, "update_rule", 'default') == 'comp':
                    # comprehensively consider both current value and child value
                    # this method will avoid A(1.0)->B(1.0)->C(1.0), because partial reasoning cannot be considered as 1.0
                    node.value = (node.value + sum(child.value * child.visits for child in node.children) / sum(child.visits for child in node.children) ) / 2
                # node.value = (node.visits * node.value + child_mean_value) / (node.visits + 1)
                # formula in https://arxiv.org/pdf/2406.07394
                # node.value = 0.5 * (node.value + max([child.value for child in node.children]))
                # node.value = 0.5 * (node.value + sum(child.value * child.visits for child in node.children) / \
                    # sum(child.visits for child in node.children))
            # node.value += value if not node.children else max([child.value for child in node.children])
            # node.visits += 1
            node = node.parent
    
    def base_terminate(self, node):
        if not node.children:
            return node.is_completed
        state = True 
        for child in node.children:
            child_state = self.base_terminate(child)
            state = state and child_state
            # state = state and self.base_terminate(child)
            if state == False:
                return False
        # node.is_completed = state

        return state

    def is_terminated(self):
        base_rule = self.base_terminate(self.root)
        if base_rule:
            return True
        correct_leaves = self.obtain_correct_leaves()
        # incorrect_leaves = self.obtain_incorrect_leaves()
        if hasattr(self.config.terminate, 'correct_nodes') and len(correct_leaves) >= self.config.terminate.correct_nodes:
            # note: this condition never satisfies during inference, because inference will seldom has correct leaves
            return True
        if self.terminate_rule == "depth":
            # depth = self.root.max_depth()
            depth = self.max_depth(self.root)
            if depth >= self.config.terminate.max_depth:
                return True 
        elif self.terminate_rule == "count":
            # count = self.root.total_node()
            count = self.total_node(self.root)
            if count >= self.config.terminate.max_nodes:
                return True 
        
        return False
    
    def expand_node(self, node: MedMCTSNode, max_children=3, bear_ratio=0.9, low_gate=0.3, **sampling_params):
        # logger.debug(f"Expanding Node: {node}")
        if node.is_completed:
            return
        if node.children:
            return 
        
        if node.value >= bear_ratio:
            max_children = 1
        
        sampling_params['n'] = max_children

        outputs = self.step_observation(node, **sampling_params)
        if all([x == '' for x in outputs]):
            
        #     # outputs = self.step_observation(node, **sampling_params)
            # node.reasoning_step += "\nThe answer is "
            outputs = self.solver.infer_answer([node.problem], [node.obtain_reasoning_steps()[0]], ['A', 'B', 'C', 'D'], add_previous_output=False)
            
        #     # sampling_params['n'] = 1
        #     # sampling_params
        #     # outputs = self.step_observation(node, **sampling_params)
            logger.debug(f'Expanding node that ends with no trigger')
            logger.debug(f'outputs: {outputs}')
        
        for i, output in enumerate(outputs):
            if output == '':
                continue
            # output = output
            if node.reasoning_step.endswith(" ") or node.reasoning_step.endswith("\n"):
                output = output.strip()
            new_node = LSMCTSNode(node.problem, output, i, parent=node, ground_truth=node.ground_truth)
            node.add_child(new_node)
    
    def step_observation(self, node: LSMCTSNode, **sampling_params):
        if not self.whether_refine:
            output = self.normal_reasoning(node, **sampling_params)
        else:
            
            if node.visits == 0:
                # first time to expand the node, maybe the root
                output = self.normal_reasoning(node, **sampling_params)
            else:
                if ("The answer is" in node.obtain_reasoning_steps()[0][-sampling_params['max_tokens']:] or "the answer is" in node.obtain_reasoning_steps()[0][-sampling_params['max_tokens']:]) and node.value < self.low_gate:
                    output = self.refine_reasoning(node, **sampling_params)
                # if node.value < node.parent.value:
                #     output = self.refine_reasoning(node, **sampling_params)
                #     # if node.depth >= self.config.expand.refine_start_depth:
                #     #     # a very bad node, need to refine
                #     #     # has already enough reasoning steps
                #     #     output = self.refine_reasoning(node, **sampling_params)
                #     # else:
                #     #     output = self.normal_reasoning(node, **sampling_params)
                #     # pass
                # elif ("The answer is" in node.obtain_reasoning_steps()[0] or "the answer is" in node.obtain_reasoning_steps()[0]) and node.value < self.low_gate:
                #     output = self.refine_reasoning(node, **sampling_params)
                else:
                    output = self.normal_reasoning(node, **sampling_params)
        
        # output: [N]
        return output
        
    
    def normal_reasoning(self, node: LSMCTSNode, **sampling_params):
        # max_tokens = sampling_params.pop("max_tokens", 1024)
        # max_tokens = 64
        if self.training:
            prompt = mcts_prompts['ls_prefix'].format(problem=node.problem)
            few_shot_prompt = "\n\n".join(mcts_ls_examples[:self.few_shot])
            cur_prompt = "Example: " + few_shot_prompt + "\n\n" + prompt
        else:
            cur_prompt = node.problem
        cur_prompt = chat_prompt(cur_prompt, tokenizer=self.model_server.tokenizer)[0]
        cur_prompt += node.obtain_reasoning_steps()[0]
        # if isinstance(self.autostep, int):
        #     # max_tokens = self.autostep
        #     sampling_params['max_tokens'] = self.autostep
        # elif isinstance(self.autostep, str):
        #     sampling_params['stop'] = [f"Step {i}:" for i in range(1, 100)]
        reasoning_steps = self.model_server([cur_prompt], wrap_chat=False, **sampling_params)[0]
        # new_steps = []
        # for step in reasoning_steps:
        #     if step.endswith("."):
        #         new_steps.append(step)
        #         continue
        #     cur_prompt += step
        #     new_step = self.model_server([cur_prompt], wrap_chat=False, stop=[self.model_server.tokenizer.eos_token, "\n\n"], max_tokens=max_tokens, n=1, temperature=sampling_params.get("temperature", 1))[0][0]
        #     step += new_step + "\n\n"
        #     new_steps.append(step)
        return reasoning_steps
        
        
    
    
    def refine_reasoning(self, node: LSMCTSNode, **sampling_params):
        trigger = "The answer is"
        refine_prompt = " Wait, this answer is incorrect. Let me "
        # self.refine_reasoning.refine_prompt = refine_prompt
        reasoning_history = node.obtain_reasoning_steps()[0]
        if trigger in reasoning_history or trigger.lower() in reasoning_history:
            logger.info("Refine wrong answer.")
        # max_tokens = sampling_params.pop("max_tokens", 1024)
        # max_tokens = 64
        if self.training:
            prompt = mcts_prompts['ls_refine_prefix'].format(problem=node.problem, wrong=reasoning_history, refine_prompt=refine_prompt)
            few_shot_prompt = "\n\n".join(mcts_ls_examples[:self.few_shot])
            cur_prompt = "Example: " + few_shot_prompt + "\n\n" + prompt
        else:
            cur_prompt = node.problem
            
        cur_prompt = chat_prompt(cur_prompt, tokenizer=self.model_server.tokenizer)[0]
        # cur_prompt += reasoning_history
        refine_answer_nodes = False
        if trigger in reasoning_history or trigger.lower() in reasoning_history:
            cur_prompt += refine_prompt
            refine_answer_nodes = True
        # if isinstance(self.autostep, int):
        #     # max_tokens = self.autostep
        #     sampling_params['max_tokens'] = self.autostep
        # elif isinstance(self.autostep, str):
        #     sampling_params['stop'] = [f"Step {i}:" for i in range(1, 100)]
        reasoning_steps = self.model_server([cur_prompt], wrap_chat=False, **sampling_params)
        new_steps = []
        for x in reasoning_steps[0]:
            if x == '':
                new_steps.append(x)
            else:
                new_steps.append(refine_prompt + x.strip() if refine_answer_nodes else x)
        # reasoning_steps = [refine_prompt + x.strip() if refine_answer_nodes else x for x in reasoning_steps[0] ]
        # return reasoning_steps
        return new_steps
    
    def rollout(self, node: LSMCTSNode, simu_func: Callable[[LSMCTSNode], tuple[float, float]], simulation=20):
        # logger.debug(f"Rollout Node: {node}")
        trigger = "The answer is"
        reasoning_history = node.obtain_reasoning_steps()[0]
        if trigger in reasoning_history or trigger.lower() in reasoning_history:
            if self.value_function is None:
                value = node.eval_node(None)
            else:
                if node.parent == self.root:
                    value = 0, 0
                    return value
                value = simu_func(node)
            return value 
        
        value = simu_func(node, simulation)
        return value 
    
    # def post_process(self, node: LSMCTSNode, value_function=None, **sampling_params):
    #     if node.is_completed:
    #         return 
        
    #     if not node.children:
    #         trigger = "The answer is"
    #         reasoning_history = node.obtain_reasoning_steps()[0]
    #         if (trigger in reasoning_history or trigger.lower() in reasoning_history) and reasoning_history.endswith("."):
    #             if node.visits == 0:
    #                 # not yet visited, eval and back propagate
    #                 if not node.reasoning_step.endswith("."):
    #                     if self.training:
    #                         prompt = mcts_prompts['ls_prefix'].format(problem=node.problem)
    #                         few_shot_prompt = "\n\n".join(mcts_ls_examples[:self.few_shot])
    #                         cur_prompt = "Example: " + few_shot_prompt + "\n\n" + prompt
    #                     else:
    #                         cur_prompt = node.problem
    #                     cur_prompt = chat_prompt(cur_prompt, tokenizer=self.model_server.tokenizer)[0]
    #                     cur_prompt += reasoning_history
    #                     sampling_params['n'] = 1
    #                     sampling_params['max_tokens'] = 1024
    #                     reasoning_steps = self.model_server([cur_prompt], wrap_chat=False, stop=[self.model_server.tokenizer.eos_token], **sampling_params)
    #                     reasoning_step = reasoning_steps[0][0]
    #                     node.reasoning_step + reasoning_step
    #                 value, simu_value = node.eval_node(value_function, self.training)
    #                 self.back_propagate(node, float(value), simu_value)
    #             # else:
    #             #     node.eval_node(value_function)
    #             return
            
    #         if self.training:
    #             prompt = mcts_prompts['ls_prefix'].format(problem=node.problem)
    #             # few_shot_prompt = "\n\n".join(mcts_ls_examples[:self.few_shot])
    #             cur_prompt =  prompt
    #         else:
    #             cur_prompt = node.problem
            
    #         cur_prompt = chat_prompt(cur_prompt, tokenizer=self.model_server.tokenizer)[0]
    #         cur_prompt += reasoning_history
    #         sampling_params['n'] = 1
    #         sampling_params['max_tokens'] = 1024
    #         reasoning_steps = self.model_server([cur_prompt], wrap_chat=False, stop=[self.model_server.tokenizer.eos_token], **sampling_params)
    #         reasoning_step = reasoning_steps[0][0]
    #         new_node = LSMCTSNode(node.problem, reasoning_step, 0, parent=node, ground_truth=node.ground_truth)
    #         node.add_child(new_node)
    #         value, simu_value = new_node.eval_node(value_function, self.training)
    #         self.back_propagate(new_node, float(value), simu_value)
    #     else:
    #         for child in node.children:
    #             self.post_process(child,  value_function=value_function, **sampling_params)
    def post_process(self, **sampling_params):
        # obtain all leaves
        leaves = self.obtain_leaves(self.root)
        # for all leaves, verdict whether it has verified prompt
        for leaf in leaves:
            reasoning_history = leaf.obtain_reasoning_steps()[0]
            if extract_template(reasoning_history, 'answer') is not None and ". Wait," in reasoning_history or extract_template(reasoning_history, 'answer') is None:
                # refine node, not yet completed or not yet completed normal reasoning node
                cur_prompt = leaf.problem 
                cur_prompt = chat_prompt(cur_prompt, tokenizer=self.model_server.tokenizer)[0]
                cur_prompt += reasoning_history
                sampling_params['n'] = 1
                sampling_params['max_tokens'] = 1024
                reasoning_steps = self.model_server([cur_prompt], wrap_chat=False, stop=[self.model_server.tokenizer.eos_token], **sampling_params)
                reasoning_step = reasoning_steps[0][0]
                if reasoning_step is not None:
                    new_node = LSMCTSNode(leaf.problem, reasoning_step, 0, parent=leaf, ground_truth=leaf.ground_truth)
                    leaf.add_child(new_node)
                    value, simu_value = new_node.eval_node(self.value_function, self.training)
                    self.back_propagate(new_node, float(value), simu_value)
            else:
                continue
                
    
    def run(self, **sampling_params):
        iter_time = 1
        if isinstance(self.autostep, int):
            sampling_params['max_tokens'] = self.autostep
        elif isinstance(self.autostep, str):
            sampling_params['stop'] = [f"Step {i}:" for i in range(1, 100)]
        while not self.is_terminated():
            return_code = self.run_one_iter(**sampling_params)
            if return_code == -1:
                logger.debug("Terminate due to -1 return code")
                break
            iter_time += 1

        # post process the tree
        # pdb.set_trace()
        # if self.value_function is None:
        #     value_function = None 
        # else:
        #     value_function = self.value_func_simulation
        if not self.training:
            self.post_process(**sampling_params)
        return self.root  
            

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from peft import PeftModel
    torch.set_default_device("cuda")
    model_path = '/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct'
    # model_path = '/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3.1-8B-Instruct'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = set_tokenizer(tokenizer)
    # data = client.read("s3://syj_test/datasets/medical_train/pubhealth.json")
    # data = client.read("s3://syj_test/datasets/medical_test/MedQA_cot.json")
    data = client.read("s3://syj_test/datasets/medical_train/mmed_en_train.json")
    # for i in range(15):
    
    i = 10
    item = data[i]
    logger.info(item)
    # logger.info(f"Correct answer: {item['answer_idx']}")
    if 'answer_idx' not in item:
        item['answer_idx'] = item['eval']['answer']
    # # print(node)
    config = client.read("Evol_Instruct/config/new_refine.json")
    config = MCTSConfig(config)
    # config.max_children=6
    # model_base = LlamaForValueFunction.from_pretrained(
    #     "/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct",
    #     num_labels=1
    # )
    # value_model = PeftModel.from_pretrained(model_base, "/mnt/petrelfs/jiangshuyang.p//checkpoints/llama38b_mcts_vllm_mmed_en_train_all_trial6/sft_combined_1-llama3-8b-r16a32-1epoch-VALUE-ITER1")
    # value_model = value_model.merge_and_unload().to(torch.float16)
    # value_model = model_base
    value_model = None
    server = VLLMServer(url="http://10.140.1.163:10002", model=model_path, tokenizer=tokenizer, offline=False)
    # server = VLLMServer(url="http://10.140.1.163:10002", model=model_path, tokenizer=tokenizer, offline=True, lora_path='/mnt/petrelfs/jiangshuyang.p//checkpoints/llama38b_mcts_vllm_mmed_en_train_all_trial6/sft_combined_1-llama3-8b-r16a32-1epoch-SFT-ITER1', gpu_memory_usage=0.45)
    # config.max_children = 6
    tree = LSMCTS(item, model_server=server, config=config, value_function=value_model, training=True)
    tokenizer = tree.tokenizer 
    start = time.time()
    # node = tree.tot_inference()
    root = tree.run(temperature=1)
    # node, node_steps = tree.inference("vote-sc")
    # exit(0)
    # correct_leaves = tree.obtain_correct_leaves()
    # repeat_try = 0
    # while not correct_leaves and repeat_try > 0:
    #     tree = MCTS(item, model_server=server, config=config, value_function=value_model)
    #     root = tree.run()
    #     correct_leaves = tree.obtain_correct_leaves()
    #     repeat_try -= 1
    end = time.time()
        
    # pdb.set_trace()
    client.write("", "debug.log", mode='w')
    fp = open("debug.log", 'a')
    leaves = tree.obtain_leaves(root)
    correct_leaves = tree.obtain_correct_leaves()
    incorrect_leaves = tree.obtain_incorrect_leaves()
    # node, node_steps = tree.inference("vote-sc")
    # print("Vote SC:", node_steps, node.value_chain(), file=fp)
    # node, node_steps = tree.inference("vote-mean")
    
    # print("Vote Sum:", node_steps, node.value_chain(), file=fp)
    # node, node_steps = tree.inference("vote-sum")

    # print("Vote-mean:", node_steps, node.value_chain(), file=fp)
    # node, node_steps = tree.inference("level_max")

    # print("level-max:", node_steps, node.value_chain(), file=fp)
    # print("*" * 100, file=fp)
    # node = tree.tot_inference()
    # print(node.obtain_reasoning_steps()[0], file=fp)
    # exit(0)
    # reasoning_sentences = 
    
    # reasoning_sentences = tree.construct_output_trace(correct_leaves)
    # for sentence in reasoning_sentences:
    #     print("Constructed new sentence:", sentence, file=open("debug.log", 'a'))
    #     print("*" * 100, file=open("debug.log", 'a'))
    # pdb.set_trace()
    # pdb.set_trace()
    # client.write("", "debug.log", mode='w')
    for leaf in leaves:
        # with open("debug.log")
        print(leaf.value_chain(), file=fp)
        print(leaf.obtain_reasoning_steps()[0], file=fp)
        print("*" * 80, file=fp)

    # correct_leaves = tree.obtain_correct_leaves()
    # for leaf in correct_leaves:
    #     print(leaf.obtain_reasoning_steps()[0])
    #     print("*" * 80)
    logger.info(f"Time cost: {end - start}")