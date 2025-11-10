from collections import Counter, defaultdict
from copy import deepcopy
import math
import time 
from typing import Any, Callable, Union, Optional

import numpy as np
import re
import torch

from queue import PriorityQueue
from Evol_Instruct.MCTS.utils import extract_template, parse_action_params
from Evol_Instruct.actions.base_action import Finish, Reflect, Think, Reason, Refine
from Evol_Instruct.evaluation.generate_utils import set_tokenizer
# from Evol_Instruct.models.modeling_value_llama import LlamaForValueFunction

from Evol_Instruct.utils.utils import compute_weighted_values, timeout_retry_decorator, LogitBiasProcess
from Evol_Instruct.models.vllm_support import VLLMServer, chat_prompt
from Evol_Instruct.prompts.prompt_template import mcts_prompts, search_prompts
from transformers import LogitsProcessorList
from Evol_Instruct import client, logger
from Evol_Instruct.MCTS.tree_node import MedMCTSNode
from Evol_Instruct.actions.actions_register import ACTIONS_REGISTRY
from Evol_Instruct.actions.base_action import BaseAction, base_actions
from Evol_Instruct.actions.medrag import Medrag, RAGReason
# from Evol_Instruct.actions.hint import Hint
from Evol_Instruct.MCTS.tree_register import register_tree, tree_registry
import pickle
import pdb

class MCTSConfig:
    def __init__(self, data):
        # self.data = data
        for key, value in data.items():
            if isinstance(value, dict):
                # 如果值是字典，则递归转换
                value = MCTSConfig(value)
            setattr(self, key, value)
    
    def __setattr__(self, name: str, value: Any) -> None:
        # setattr(self, name, value)
        self.__dict__[name] = value
        # pass
    
    def to_json(self):
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, MCTSConfig):
                data[key] = value.to_json()
            else:
                data[key] = value
        return data


                
@register_tree("MCTS")
class MCTS:
    def __init__(self, item, model_server: VLLMServer, config, value_function=None, training=True, first_round=False) -> None:
        if isinstance(item, dict):
            node = MedMCTSNode(item['conversations'][0]['value'].strip("\"").strip("\n"), 
                            reasoning_step=mcts_prompts['break_down'], 
                            index=0, 
                            ground_truth=item['answer_idx'],
                            refine_limit=config.refine_limit)
        elif isinstance(item, str):
            node = MedMCTSNode(item, reasoning_step=mcts_prompts['break_down'], index=0,
                               ground_truth="", refine_limit=config.refine_limit)
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
        self.terminate = False
        self.finish_nodes = []
        self.first_round = first_round
        if isinstance(item, dict):
            item_id = item['id']
        else:
            item_id = "not given"
        logger.info(f"MCTS init with {self.first_round} first round, {item_id}")
        self.post_init()
    
    def is_reflect_node(self, node):
        condition1 = node.value < 0.5 and node.visits > 0
        condition1 = node.visits > 0 and (node.value < node.parent.value) and node.depth > 1
        condition2 = node.type != 'Medrag'
        # condition3 = extract_template(node.reasoning_step, 'answer') is not None
        return condition1 and condition2
        
    def post_init(self):
        
        if getattr(self.config.expand, "actions", ["base"]) == ["base"]:
            # load all base actions
            all_actions = []

            all_actions = {action_name: action() for action, action_type, action_name in ACTIONS_REGISTRY if action_type == 'base'}
            # all_actions = [action() for action, action_type in ACTIONS_REGISTRY if action_type == 'base']
            self.actions = all_actions
        else:
            all_actions = {}
            for action in getattr(self.config.expand, "actions", []):
                try:    
                    if action == 'base':
                        all_actions = {action_name: action() for action, action_type, action_name in ACTIONS_REGISTRY if action_type == 'base'}
                    else:
                        params = parse_action_params(action, self.config)
                        all_actions[action] = eval(action, globals())(**params)
                except Exception as e:
                    print(e)
                    print(f"{action} is not predefined in the action space")
            self.actions = all_actions 
        self.think_action = Think({"Reason": self.actions['Reason'],
                                   "Finish": self.actions['Finish']})
        self.base_actions = base_actions
        self.reflect_action = Reflect()
            
    
    def direct_infer_simulation(self, node: MedMCTSNode, simulation_times):
        simulation_times = max(simulation_times // (node.depth + 1), 5)
        if self.first_round:
            
            prompt = node.build_simu_prompt()
            
            answers = self.model_server([prompt], n=simulation_times, temperature=0.6,top_p=0.9)
        else:
            prompt = chat_prompt([node.problem], self.tokenizer)[0] + node.obtain_reasoning_steps()[0]
            answers = self.model_server([prompt], n=simulation_times, wrap_chat=False, temperature=1.0, top_p=0.9, max_tokens=1536)
        text = answers[0]
        predict_idx = [extract_template(x, 'answer') for x in text]
        is_correct = [node.is_correct(x, node.ground_truth) if x is not None else False for x in predict_idx]
        # is_correct = [x in node.ground_truth if x is not None else False for x in predict_idx ]
        node.simulations = [int(x) for x in is_correct]
        accuracy = sum(is_correct) / len(is_correct)
        return accuracy, accuracy

    def value_func_simulation(self, node: MedMCTSNode, *args, **kwargs):
        trajectory = node.obtain_reasoning_steps()[0]
        # if node.type == 'Finish':
        #     trajectory = node.obtain_reasoning_steps()[0]
        # else:
        #     trajectory = node.obtain_reasoning_steps()[0]
        
        # value function will takes the trajectoy as input and output value 
        conversation = [
            {"role": "user", "content": node.problem},
            # {"role": "assistant", "content": trajectory}
        ]
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) + trajectory + "\n\n"
        # assistant_content_index = text.index(trajectory)
        # assistant_content_length = len(trajectory)
        # text = text[:assistant_content_index + assistant_content_length] + "\n\n" + text[assistant_content_index + assistant_content_length:]
        # eos_len = len(self.tokenizer.eos_token)
        # text = text[:-eos_len] + "\n\n" + self.tokenizer.eos_token
        
        tokens = self.tokenizer(text, return_tensors='pt', padding=True)
        tokens = {k: v.to(self.value_function.device) for k, v in tokens.items()}
        value = self.value_function(**tokens)
        value_model_score = value.item()
        # tokens = self.tokenizer(trajectory, return_tensors='pt', padding=True)
        # with torch.inference_mode():
        #     value = self.value_function(**tokens)

        # # if isinstance(self.value_function, LlamaForSequenceClassification)
        # value_model_score = torch.sigmoid(value[0]).item()
        # value_model_score = value[0].item()
        simulation_score = None
        if self.training and node.type == 'Finish':
            simulation_score = node.eval_node()[1]
        #     simulation_score, _ = self.direct_infer_simulation(node, self.config.simulation)
            # value_model_score = (value_model_score + simulation_score) / 2
        return value_model_score, simulation_score

    def max_depth(self, node: MedMCTSNode):
        # compute the max depth of the whole tree
        all_depths = []
        for child in node.children:
            all_depths.append(self.max_depth(child))
        if all_depths:
            return max(all_depths)
        else:
            return node.depth
    
    def total_node(self, node: MedMCTSNode):
        all_nodes = 1
        for child in node.children:
            all_nodes += self.total_node(child)
        return all_nodes
    
    def base_terminate(self, node):
        # pdb.set_trace()
        # terminate if all leaf nodes reach 'Finish' state

        if not node.children:
            # leaf node, 
            return node.type == 'Finish' and node.is_completed
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
            # note: this condition never satisfies during inference, because inference will never set correct nodes
            return True
        # leaves = [leaf for leaf in self.obtain_leaves(self.root) if leaf.type == 'Finish']
        # if not self.training and (len(self.finish_nodes) < getattr(self.config.terminate, "least_leaves", 16)):
        #     return False
        if self.terminate_rule == "depth":
            # depth = self.root.max_depth()
            depth = self.max_depth(self.root)
            if depth >= self.config.terminate.max_depth:
                return True 
        elif self.terminate_rule == "count":
            # count = self.root.total_node()
            count = self.total_node(self.root)
            if count >= self.config.terminate.max_nodes:
                # finish_nodes = [node for node in self.obtain_leaves(self.root) if node.type == 'Finish']
                # if len(finish_nodes) == 0 and :
                #     self.config.terminate.max_nodes += 10
                #     return False 
                return True 
        
        return False
                

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
        if node.type == 'Finish' and node.value > 0:
            # has been rollout before
            return
        if node.visits == 0 and node.depth == 0:
            # root
            # logger.debug("Choose to expand due to root node")
            self.expand_node(node, 
                            max_children=self.config.expand.max_children,
                            bear_ratio=self.bear_ratio,
                            low_gate=self.low_gate, **sampling_params)
        # elif node.visits == 0 and node.depth > 0:
        #     if self.value_function is not None:
        #         # inference mode
        #         value, simulation_score = self.rollout(node, self.value_func_simulation)
        #     else:
        #         # pdb.set_trace()
        #         value, simulation_score = self.rollout(node, self.direct_infer_simulation, simulation=self.config.simulation)
        #     # logger.debug(f"Choose to rollout due to leaf node not visited, Value: {value}")
        #     self.back_propagate(node,value, simulation_score)
        elif node.visits > 0 and node.depth > 0:
            # logger.debug(f"Choose to expand {node} due to leaf node visited")
            # pdb.set_trace()
            self.expand_node(node,
                            #  lora_request=self.lora_request,
                             max_children=self.config.expand.max_children,
                             bear_ratio=self.bear_ratio,
                             low_gate=self.low_gate,  **sampling_params)
        # else:
        return 0
    
    # @timeout_retry_decorator(timeout_duration=600)
    def run(self, **sampling_params):
        if len(self.model_server.tokenizer.encode(self.root.problem)) > 8192:
            return None
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
        if self.value_function is None:
            value_function = None 
        else:
            value_function = self.value_func_simulation

        self.post_process(self.root, value_function=value_function, **sampling_params)
        return self.root   

    def inference(self, select_rule='max', **kwargs):
        # after run the MCTS, select the highest value node 
        # node = self.root 
        # while node.children:
        #     child_node = self.select_child(node, self.constant, training=False)
        #     # if child_node is None:
        #     #     answer_prompt = 
        #     node = child_node
        
        # reasoning_path = node.obtain_reasoning_steps()[0]
        # leaves = self.obtain_leaves(self.root)
        leaves = self.finish_nodes
        if select_rule == 'max':
            leaf = None
            cur_value = -1
            for single_leaf in leaves:
                if single_leaf.value > cur_value:
                    cur_value = single_leaf.value 
                    leaf = single_leaf 
            # best_node = leaf 
            # node_step = leaf.obtain_reasoning_steps()[0]
            return leaf, leaf.obtain_reasoning_steps()[0]
        elif select_rule == 'level_max':
            temp = self.root 
            # for child in temp.children:
            while temp.children:
                # child_node = self.select_child(temp, 0)
                child_node = max(temp.children, key=lambda x: x.value)
                temp = child_node 
            # best_node = temp 
            # node_step = temp.obtain_reasoning_steps()[0]
            return temp, temp.obtain_reasoning_steps()[0]

        elif select_rule.startswith('vote-') or select_rule.startswith('prm-'):
            leaves = [leaf for leaf in leaves if leaf.type == 'Finish']
            only_answer_outputs = [extract_template(leaf.reasoning_step, 'answer') for leaf in leaves]
            values = [[0] + leaf.value_trajectory for leaf in leaves] if select_rule.startswith('prm') else [leaf.value for leaf in leaves]
            max_answer, weighted_values = compute_weighted_values(only_answer_outputs, values, select_rule)
            
            # max_answer = max(weighted_values, key=weighted_values.get)
            max_weighted_value = weighted_values[max_answer]
            tie_answers = [ans for ans, val in weighted_values.items() if abs(val - max_weighted_value) < 1e-6]
            if len(tie_answers) == 1:
                # 没有平局，直接选择 max_answer 对应的节点
                # best_node = 
                best_node =  max((node for node in leaves if extract_template(node.reasoning_step, 'answer') == max_answer), key=lambda n: n.value)
            else:
                best_node = None
                best_value = float('-inf')
                for leaf in leaves:
                    if extract_template(leaf.reasoning_step, 'answer') in tie_answers and leaf.value > best_value:
                        best_value = leaf.value 
                        best_node = leaf
            return best_node, best_node.obtain_reasoning_steps()[0]
        elif select_rule == 'tot':
            node = self.tot_inference(**kwargs)
            return node, node.obtain_reasoning_steps()[0]

            
            

    def back_propagate(self, node: MedMCTSNode, value, simulation_score=None):
        node.simulation_score = simulation_score
        if node.parent.value_trajectory == []:
            node.value_trajectory.append(value)
        else:
            node.value_trajectory = node.parent.value_trajectory + [value]
        while node:

            node.visits += 1
            if not node.children:
                node.value = value if node.type != 'Finish' else simulation_score if simulation_score is not None else value
                is_completed = node.value >= self.config.expand.bear_ratio or \
                    (node.value < self.config.expand.low_gate and node.refine_cnt >= self.config.refine_limit)
                node.is_completed = is_completed if node.type == 'Finish' else False
                if simulation_score is not None:
                    node.correct = simulation_score >= self.bear_ratio
            else:
                
                if getattr(self.config, 'update_rule', 'default') == 'default':
                    child_mean_value = sum(child.value * child.visits for child in node.children) / sum(child.visits for child in node.children)
                    node.value = child_mean_value
                elif getattr(self.config, "update_rule", 'default') == 'comp':
                    # comprehensively consider both current value and child value
                    # this method will avoid A(1.0)->B(1.0)->C(1.0), because partial reasoning cannot be considered as 1.0
                    node.value = (node.value + sum(child.value * child.visits for child in node.children) / sum(child.visits for child in node.children) ) / 2
                node.is_completed = all(child.is_completed for child in node.children if child is not None)
                # node.value = (node.visits * node.value + child_mean_value) / (node.visits + 1)
                # formula in https://arxiv.org/pdf/2406.07394
                # node.value = 0.5 * (node.value + max([child.value for child in node.children]))
                # node.value = 0.5 * (node.value + sum(child.value * child.visits for child in node.children) / \
                    # sum(child.visits for child in node.children))


            node = node.parent
    
    def select_child(self, node, constant=2):
        max_ucb = -1
        return_node = None
        constant_change = getattr(self.config.expand, 'constant_change', 'constant')
        # if constant_change != 'constant':
        #     constant = eval(constant_change)
        for child in node.children:
            if self.training:
                value = child.value 
            else:
                value = np.prod(child.value_trajectory)
            if constant_change != 'constant':
                cur_constant = eval(constant_change)
            else:
                cur_constant = constant
            if child.visits == 0:
                ucb = getattr(self.config.expand, 'unvisited_ucb', math.inf)
            else:
                if child.correct and child.type == 'Finish':
                    ucb = -1
                elif child.type == 'Finish' and (child.value < self.low_gate and child.refine_cnt < self.config.refine_limit):
                    # a bad node, we can refine 
                    ucb = value + cur_constant * math.sqrt(math.log(node.visits) / child.visits)
                elif child.is_completed:
                    # a finish node
                    # either a good node 
                    # or a bad node which has undergo sufficient refinement
                    # a seemly completed finish node with refinement quota is not listed here
                    ucb = -1
                else:
                    ucb = value  + cur_constant * math.sqrt(math.log(node.visits) / child.visits)
            if ucb > max_ucb:
                if max_ucb == getattr(self.config.expand, 'unvisited_ucb', math.inf):
                    logger.info("Explore already visited nodes")
                max_ucb = ucb
                return_node = child

        return return_node

    def normal_expand_node(self, node: MedMCTSNode, max_children=3, **sampling_params):
        sampling_params['n'] = max_children
        # cur_action = 'Think'
        think_action = self.think_action
        # step_count = len(node.trace)
        if node.depth != 0:
            if node.type == 'Medrag':
                next_actions = ['Reason'] * max_children
            else:
                next_actions = think_action(node, self.model_server, **sampling_params)
        # pdb.set_trace()
        else:
            next_actions = ['Reason'] * max_children

        final_actions = []
        for action in next_actions:
            if action != '':
                final_actions.append(action)
        if not final_actions:
            final_actions = ['Reason'] * max_children
        return final_actions

    def expand_node(self, node: MedMCTSNode, max_children=3, bear_ratio=0.9,
                    low_gate=0.3, whether_expand_finish=True, defined_actions=None, **sampling_params):
        if node.is_completed and node.value < low_gate and node.refine_cnt < node.refine_limit:
            return 
        if node.children != []:
            return 
        # logger.info(f"Select node: {node.trace}")
        if node.simulation_score is not None and node.simulation_score >= bear_ratio:
            # training
            max_children = 1
            next_actions = ['Finish']
        
        else:
            if node.depth == 0:
                if "Medrag" in self.actions:
                    next_actions = ['Medrag'] * len(getattr(getattr(self.config, "Medrag", {}), "source_list", ["MedCorp"]))
                else:
                    next_actions = self.normal_expand_node(node, max_children, **sampling_params)
            else:
                # if "Reflect" in self.actions:
                #     # if node.value <= low_gate and node.visits > 0 and (node.type != 'Medrag'):
                #     if self.is_reflect_node(node):
                #         # a very bad node has already rollout, need to use reflect

                #         action = 'Reflect'
                        
                #         n = max_children
                        
                #         next_actions = [action] * n
                #         if getattr(self.config, "refine", False):
                #             self.refine_node(node)
                #     else:
                #         next_actions = self.normal_expand_node(node, max_children, **sampling_params)
                if node.type == 'Finish' and node.value < low_gate and self.config.refine_limit > 0 and self.config.refine_limit > node.refine_cnt:
                    node.type = 'Reason'
                    node.is_completed = False
                    # recursively set all parent nodes is_completed to False
                    # current = node
                    # while current.parent:
                    #     current = current.parent
                    #     current.is_completed = False
                    next_actions = ['Reflect']
                else:
                    if getattr(self.config, "refine", False):
                        self.refine_node(node)
                    next_actions = self.normal_expand_node(node, max_children, **sampling_params)
            
        
        sampling_params['n'] = 1
        # if node.value < low_gate and node.depth > 0:
        #     # a bad node, we cannot reflect, so we can only add a visits to it and do nothing
        #     node.visits += 1
        #     return 
        if defined_actions is not None:
            next_actions = defined_actions
        if "Medrag" in next_actions:
            observations = self.multisource_rag(node, next_actions, **sampling_params)
        else:
            observations = self.step_observation(node, next_actions, **sampling_params)
        
    
        for i in range(len(next_actions)):
            if observations[i] is None:
                continue
            step = observations[i].strip("\n")
            step = step.replace("<steps>", "") if step.startswith("<steps>") else step
            
            # step = step.replace("<steps>", "").strip("\n")
            if next_actions[i] == 'Finish' or next_actions[i] == 'Reflect':
                refine_incre = (1 if next_actions[i] == 'Reflect' else 0)
                self.process_answer_nodes(node, step, whether_expand_finish=True, refine_incre=refine_incre)
            else:
                # refine_incre = (1 if next_actions[i] == 'Reflect' else 0)
                if extract_template(step.strip(), 'answer') is not None:
                    next_actions[i] = 'Finish'
                new_node = MedMCTSNode(node.problem, step.strip(), i, parent=node, type=next_actions[i], ground_truth=node.ground_truth)
                node.add_child(new_node)
                if next_actions[i] == 'Finish':
                    self.finish_nodes.append(new_node)
                if self.value_function is not None:
                    # inference mode
                    value, simulation_score = self.rollout(new_node, self.value_func_simulation)
                else:
                    # pdb.set_trace()
                    value, simulation_score = self.rollout(new_node, self.direct_infer_simulation, simulation=self.config.simulation)
                self.back_propagate(new_node, value, simulation_score)
        if node.children == []:
            # exceed the limit , mark as completed
            node.is_completed = True
        return
            
    def refine_node(self, node: MedMCTSNode):
        act_class = Refine()
        temp = node
        refine_max = getattr(self.config, "refine_max", 3)
        while refine_max > 0 and temp.value <= self.low_gate:
            temp = deepcopy(node)
            new_step = act_class(node, self.model_server, few_shot=0, n=1, temperature=1)
            temp.reasoning_step += "\nWait, I have made a mistakes in the previous steps. Let me refine it.\n" + new_step[0]
            if self.value_function is not None:
                
                value, simulation_score = self.rollout(temp, self.value_func_simulation)
            else:
                value, simulation_score = self.rollout(temp, self.direct_infer_simulation, self.config.simulation)
            temp.value = value 
            refine_max -= 1
        if refine_max == 0:
            logger.info("refine failed, maintain node as its originality")
        else:
            logger.info("refine success, update the node value")
            node.reasoning_step = temp.reasoning_step

    
    
    def process_answer_nodes(self, node: MedMCTSNode, reasoning_step: str, whether_expand_finish: bool = True, refine_incre: int = 0):
        """
        Split a potentially multi‑step finish into intermediate 'Reason' nodes
        and a final node, then rollout and back‑propagate each.
        """
        # 1. split into steps
        parts = reasoning_step.split("Step")
        steps = [parts[0].strip()] + [p[4:].strip() for p in parts[1:]]
        multi = len(steps) > 1 and extract_template(steps[-1], "answer") is not None

        parent = node
        new_nodes = []

        # 2. create intermediate Reason nodes if multi‐step and allowed
        if multi and whether_expand_finish:
            for text in steps[:-1]:
                child = MedMCTSNode(
                    node.problem,
                    text,
                    len(parent.children),
                    parent=parent,
                    type="Reason" if refine_incre == 0 else 'Reflect',
                    ground_truth=node.ground_truth,
                    refine_cnt=node.refine_cnt + refine_incre
                )
                # inherit parent stats
                child.value = parent.value
                child.value_trajectory = parent.value_trajectory.copy()
                child.visits = 1
                parent.add_child(child)
                new_nodes.append(child)
                parent = child
            # account for extra nodes in termination count
            self.config.terminate.max_nodes += len(steps) - 1
            final_text = steps[-1]
        else:
            final_text = steps[0]
        
        
        # remove duplicated "The answer is" if present twice
        tmpl = "The answer is"
        ans = extract_template(final_text, "answer")
        if ans:
            idx = final_text.lower().find(tmpl.lower())
            nxt = final_text.lower().find(tmpl.lower(), idx + len(tmpl))
            if nxt != -1:
                final_text = final_text[:nxt].strip()

        node_type = "Finish" if ans else "Reason"
        final = MedMCTSNode(
            node.problem,
            final_text,
            len(parent.children),
            parent=parent,
            type=node_type,
            ground_truth=node.ground_truth,
            refine_cnt=node.refine_cnt + refine_incre
        )
        parent.add_child(final)
        # new_nodes.append(final)
        if node_type == "Finish":
            self.finish_nodes.append(final)

        # 4. rollout & back‑propagate final node

        rollout_fn = self.value_func_simulation if self.value_function else self.direct_infer_simulation
        val, sim = self.rollout(final, rollout_fn, simulation=self.config.simulation)

        self.back_propagate(final, val, sim)

        return final
    
    
    def rollout(self, node: MedMCTSNode, simu_func: Callable[[MedMCTSNode], tuple[float, float]], simulation=20):
        # based on previous reasoning steps to infer the final answer
        # infer simulation times, and return the mean accuracy
        # a node with a higher depth should use less simulation times because it has more information

        if node.type == 'Finish':
           
            # value = node.eval_node()
            if self.value_function is None:
                value = node.eval_node(None)
            else:
                value = self.value_func_simulation(node)
                # value = simu_func(node)
            if "Reflect" in self.actions:
                if value[0] < self.low_gate:
                    node.type = 'Reason'
            # self.back_propagate(node, value[0], value[1])
            return value 
        
        value = simu_func(node, simulation)
        # self.back_propagate(node, value[0], value[1])
        return value 
    
    def post_process(self, node: MedMCTSNode, value_function=None, **sampling_params):
        # once terminated, check if all leaf nodes is a Finish node, if not,
        # expand one child to obtain the Finish node
        if node.is_completed:
            return
        leaves = self.obtain_leaves(self.root)
        finish_leaves = [leaf for leaf in leaves if leaf.type == 'Finish' and (leaf.correct or (not self.training))]
        if len(finish_leaves) <= 3:
            finish_uncompleted = True
        else:
            finish_uncompleted = getattr(self.config, 'finish_uncompleted', True)
        
        correct_leaves_num = len(finish_leaves)
        leaves.sort(key=lambda x: x.value, reverse=True)
        for leaf in leaves:
            if leaf.type != 'Finish' and (finish_uncompleted or correct_leaves_num <= 3):
                action = 'Finish'

                if leaf.visits == 0:
                    value, simu_value = self.rollout(leaf, self.value_func_simulation if value_function is not None else self.direct_infer_simulation)
                    self.back_propagate(leaf, value, simu_value)
                sampling_params['n'] = 1
                action_class = self.actions['Finish']
                sampling_params['max_tokens'] = 4096
                step = action_class(leaf, self.model_server, few_shot=self.few_shot, first=self.training and self.first_round, direct_output=True, **sampling_params)
                # step = self.step_llm_action(node, action, **sampling_params)
                # reasoning_steps = self.model_server(prompt=[cur_prompt], **sampling_params)
                step = step[0]
                final_node = self.process_answer_nodes(leaf, step, )
                # if step is None: 
                #     continue
                # # step = reasoning_steps[0][0]
                # new_node = MedMCTSNode(node.problem, step.strip(), 0, parent=leaf, type=action, ground_truth=node.ground_truth)
                # leaf.add_child(new_node)
                # value, simu_value = new_node.eval_node(value_function, training=self.training)
                # # if simu_value == 1
                value = final_node.value
                if abs(value - 1) < 1e-6:
                    correct_leaves_num += 1
                # self.back_propagate(new_node, value=float(value), simulation_score=simu_value)
            elif leaf.type == 'Finish':
                if leaf.visits == 0:
                    value, simu_value = leaf.eval_node(value_function, training=self.training)
                    self.back_propagate(leaf, value=float(value), simulation_score=simu_value)
        return
                    
        
    def obtain_leaves(self, root: MedMCTSNode):
        if not root.children:
            if root is None:
                return []
            return [root]

        leaves = []
        for child in root.children:
            leaves += self.obtain_leaves(child)
        return leaves
    
    def obtain_correct_leaves(self):
        # if not hasattr(self, "leaves"):
            
        self.leaves = self.obtain_leaves(self.root)

        # pdb.set_trace()
        correct_leaves = list(set([leaf for leaf in self.leaves if leaf.correct]))

        return correct_leaves

    def obtain_incorrect_leaves(self):
        # if not hasattr(self, "leaves"):
        self.leaves = self.obtain_leaves(self.root)
        incorrect_leaves = list(set([leaf for leaf in self.leaves if leaf.correct == False and leaf.type == 'Finish']))

        return incorrect_leaves
    
    
    def step_observation(self, node: MedMCTSNode, action_list: list[str], mcts_inference=False, **sampling_params):
        observations = [None for _ in range(len(action_list))]
        # obtain action_list count
        action_count = Counter(action_list)
        action_index = defaultdict(list)
        for i, action in enumerate(action_list):
            action_index[action].append(i)
        for action, index in action_index.items():
            count = action_count[action]
            sampling_params['n'] = count
            if action == 'Reflect':
                output = self.reflect_action(node, self.model_server, few_shot=0, **sampling_params)
            else:
                action_class = self.actions.get(action, None)


                if action_class.action_name == 'Finish':
                    output = action_class(node, self.model_server, few_shot=self.few_shot, first=self.training and self.first_round, direct_output=True, **sampling_params)
                else:
                    output = action_class(node, self.model_server, few_shot=self.few_shot, first=self.training and self.first_round, **sampling_params)
            # pdb.set_trace()
            
            for i in range(len(output)):
                observations[index[i]] = output[i].strip() if output[i] is not None else None

        return observations
    
    def construct_output_trace(self, nodes: list[MedMCTSNode]):
        
        prompts = []
        for node in nodes:
            reasoning_steps = node.obtain_reasoning_steps()[0]
            prompt = f"You are a professional assistant. You will be given a medical problem and a step-wise solution generated by a student. Your task is to make the step-wise solution more expertised and fluent but maintain the overall meaning. For some draft words, please remember to remove them. Also do not change the template 'The answer is [A or B or C or D or E].' in the last step because this is important for me to extract the answer. Do not change the answer even if you think the answer is incorrect. Remember not to remove retrieved documents, as they are important for inferring to the final answer.\n\nProblem: {node.problem}\nStep-wise solution: {reasoning_steps}"
            prompt = chat_prompt([prompt], tokenizer=self.model_server.tokenizer)[0] + "I'll revise the step-wise solution to make it more expertized and fluent while maintaining the overall meaning. Here's the revised solution:\n\n"
        # + "The re-organized steps are as follow:\n<answer>"
            prompts.append(prompt)
        
        output = self.model_server(prompts, wrap_chat=False, n=1, temperature=1, 
                                   stop=["</answer>", self.model_server.tokenizer.eos_token])
        output = [each[0].strip() for each in output] 
        new_output = []
        for each in output:
            match = re.search(r'<steps>(.*?)</steps>', each)
            if match:
                new_output.append(match.group(1))
            else:
                new_output.append(each)
        trigger = "The answer is "
        output = [each[:each.find(trigger) + len(trigger) + 1] + "." for each in new_output]
        return output

    
    def derive_answer(self, node: MedMCTSNode, **sampling_params):
        action_class = Finish()
        sampling_params['n'] = 1
        step = action_class(node, self.model_server, few_shot=0, first=self.training and self.first_round, direct_output=True, **sampling_params)

        step = step[0]

        new_node = MedMCTSNode(node.problem, step.strip(), 0, parent=node, type='Finish', ground_truth=node.ground_truth, refine_cnt =node.refine_cnt)
        node.add_child(new_node)
    

def print_node(node: MedMCTSNode):
    print(node)
    print("*" * 80)
    
    for child in node.children:
        print_node(child)  
