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
from Evol_Instruct.utils.utils import compute_weighted_values
from Evol_Instruct.models.vllm_support import VLLMServer, chat_prompt
from Evol_Instruct.prompts.prompt_template import mcts_prompts, search_prompts
from Evol_Instruct import client, logger
from Evol_Instruct.MCTS.tree_node import MedMCTSNode
from Evol_Instruct.actions.actions_register import ACTIONS_REGISTRY
from Evol_Instruct.actions.base_action import BaseAction, base_actions
from Evol_Instruct.actions.medrag import Medrag, RAGReason
# from Evol_Instruct.actions.hint import Hint
from Evol_Instruct.MCTS.tree_register import register_tree, tree_registry
from scipy.stats import gmean
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
                            ground_truth=item['answer_idx'] if training else "")
        elif isinstance(item, str):
            node = MedMCTSNode(item, reasoning_step=mcts_prompts['break_down'], index=0,
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
        self.terminate = False
        self.finish_nodes = []
        self.first_round = first_round
        logger.info(f"MCTS init with {self.first_round} first round")
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
            
    
    def direct_infer_simulation(self, node: MedMCTSNode, simulation_times):
        prompt = node.build_simu_prompt()
        simulation_times = max(simulation_times // (node.depth + 1), 5)
        answers = self.model_server([prompt], system='You are a helpful assistant.', n=simulation_times)
        # answers = vllm_clean_generate(self.model, prompts=[prompt], 
        #                               system='You are a helpful assistant.', 
        #                               n=simulation_times,
        #                               lora_request=self.lora_request)
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
            {"role": "assistant", "content": trajectory}
        ]
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        # assistant_content_index = text.index(trajectory)
        # assistant_content_length = len(trajectory)
        # text = text[:assistant_content_index + assistant_content_length] + "\n\n" + text[assistant_content_index + assistant_content_length:]
        eos_len = len(self.tokenizer.eos_token)
        text = text[:-eos_len] + "\n\n" + self.tokenizer.eos_token
        
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
        if not self.training and (len(self.finish_nodes) < getattr(self.config.terminate, "least_leaves", 16)):
            return False
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
            # logger.debug("Choose to expand due to leaf node visited")
            # pdb.set_trace()
            self.expand_node(node,
                            #  lora_request=self.lora_request,
                             max_children=self.config.expand.max_children,
                             bear_ratio=self.bear_ratio,
                             low_gate=self.low_gate,  **sampling_params)
        # else:
        return 0
       
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
            if node.type == 'Finish':
                # this is the finish node, and this trajectory is never explored
                # set its father nodes to be completed if its father has only one child
                node.is_completed = True
                if simulation_score is not None:
                    node.correct = simulation_score >= self.bear_ratio
                    # node.incorrect = simulation_score <= self.low_gate
                # node.correct = simulation_score >= self.bear_ratio if simulation_score is not None else False
            else:
                if node.children:
                    node.is_completed = all(child.is_completed for child in node.children if child is not None)
            # node.parent.is_completed = all(child.is_completed for child in node.parent.children)
            node.visits += 1
            if not node.children:
                node.value = value if node.type != 'Finish' else simulation_score if simulation_score is not None else value
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
            if child.is_completed:
                # all its children are explored and hence it is completed
                # choose not to explore this trace
                if cur_constant > 0:
                    ucb = -1
                
                else:
                    
                    ucb = value  + cur_constant * math.sqrt(math.log(node.visits) / child.visits)
            elif child.visits == 0:
                ucb = getattr(self.config.expand, 'unvisited_ucb', math.inf)
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
        if node.is_completed:
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
                if "Reflect" in self.actions:
                    # if node.value <= low_gate and node.visits > 0 and (node.type != 'Medrag'):
                    if self.is_reflect_node(node):
                        # a very bad node has already rollout, need to use reflect

                        action = 'Reflect'
                        
                        n = max_children
                        
                        next_actions = [action] * n
                        if getattr(self.config, "refine", False):
                            self.refine_node(node)
                    else:
                        next_actions = self.normal_expand_node(node, max_children, **sampling_params)
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
            if next_actions[i] == 'Finish':
                self.process_answer_nodes(node, step, whether_expand_finish=node.value >= bear_ratio)
            else:
                if extract_template(step.strip(), 'answer') is not None:
                    next_actions[i] = 'Finish'
                new_node = MedMCTSNode(node.problem, step.strip(), i, parent=node, type=next_actions[i], ground_truth=node.ground_truth,)
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

    def tot_expand_node(self, node: MedMCTSNode, max_children=3, bear_ratio=0.9, low_gate=0.3, whether_expand_finish=True, defined_actions=None, **sampling_params):
        next_actions = self.normal_expand_node(node, max_children, **sampling_params)
        observations = self.step_observation(node, next_actions, **sampling_params)
        for i in range(len(next_actions)):
            if observations[i] is None:
                continue
            step = observations[i].strip("\n")
            step = step.replace("<steps>", "") if step.startswith("<steps>") else step

            if next_actions[i] == 'Finish':
                if extract_template(step, 'answer') is None:
                    next_actions[i] = 'Reason'
                    
            else:
                if extract_template(step.strip(), 'answer') is not None:
                    next_actions[i] = 'Finish'
                
            new_node = MedMCTSNode(node.problem, step.strip(), len(node.children), parent=node, type=next_actions[i], ground_truth=node.ground_truth,)
            node.add_child(new_node)
    
    def process_answer_nodes(self, node: MedMCTSNode, reasoning_step: str, whether_expand_finish: bool=True):
        steps = reasoning_step.split("Step")
        steps = [step[4:] if i > 0 else step for i, step in enumerate(steps) ]
        if len(steps) > 1 and extract_template(steps[-1], "answer") is not None:
            # a Finish node generate more than one steps
            # generate intermediate steps 
            # pdb.set_trace()
            if whether_expand_finish:
                temp = node
                for each_step in steps[:-1]:
                    new_node = MedMCTSNode(node.problem, each_step.strip(), len(temp.children), parent=temp, type='Reason', ground_truth=temp.ground_truth)
                    new_node.value = temp.value 
                    new_node.value_trajectory = temp.value_trajectory
                    new_node.visits = 1
                    temp.add_child(new_node)
                    temp = new_node 
                    
                new_node = MedMCTSNode(node.problem, steps[-1].strip(), len(temp.children), parent=temp, type='Finish', ground_truth=temp.ground_truth)
                temp.add_child(new_node)
                self.config.terminate.max_nodes += len(steps) - 1
            else:
                new_node = MedMCTSNode(node.problem, reasoning_step.strip(), len(node.children), parent=node, type='Finish', ground_truth=node.ground_truth)
                node.add_child(new_node)
            if self.value_function is not None:
                # inference mode
                value, simulation_score = self.rollout(new_node, self.value_func_simulation)
            else:
                # pdb.set_trace()
                value, simulation_score = self.rollout(new_node, self.direct_infer_simulation, simulation=self.config.simulation)
            self.finish_nodes.append(new_node)
            # generate the final answer node
        else:
            # a single step for Finish action
            target = "The answer is"
            
            if extract_template(steps[-1], "answer") is not None:
                
                first_occurence = max(steps[-1].find(target), steps[-1].find(target.lower()))
            else:
                first_occurence = -1
            if first_occurence != -1:
                second_occurence = max(steps[-1].find(target, first_occurence + len(target)),
                                       steps[-1].find(target.lower(), first_occurence + len(target)))
                if second_occurence != -1:
                    steps[-1] = steps[-1][:second_occurence].strip("\n")
                else:
                    steps[-1] = steps[-1].strip("\n")
                node_type = 'Finish'
                
            else:
                node_type = 'Reason'
            new_node = MedMCTSNode(node.problem, steps[-1].strip(), len(node.children), parent=node, type=node_type, ground_truth=node.ground_truth)
            node.add_child(new_node)
            if node_type == 'Finish':
                self.finish_nodes.append(new_node)
            if self.value_function is not None:
                # inference mode
                value, simulation_score = self.rollout(new_node, self.value_func_simulation)
            else:
                # pdb.set_trace()
                value, simulation_score = self.rollout(new_node, self.direct_infer_simulation, simulation=self.config.simulation)
        
        self.back_propagate(new_node, value, simulation_score)
        return new_node
    
    
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
                step = action_class(leaf, self.model_server, few_shot=self.few_shot, first=self.training and self.first_round, direct_output=True, **sampling_params)
                # step = self.step_llm_action(node, action, **sampling_params)
                # reasoning_steps = self.model_server(prompt=[cur_prompt], **sampling_params)
                step = step[0]
                
                # step = reasoning_steps[0][0]
                new_node = MedMCTSNode(node.problem, step.strip(), 0, parent=leaf, type=action, ground_truth=node.ground_truth)
                leaf.add_child(new_node)
                value, simu_value = new_node.eval_node(value_function, training=self.training)
                # if simu_value == 1
                if abs(value - 1) < 1e-6:
                    correct_leaves_num += 1
                self.back_propagate(new_node, value=float(value), simulation_score=simu_value)
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

            action_class = self.actions[action]
            # output = action_class(node, self.model_server, few_shot=self.few_shot, first=self.training, **sampling_params)
            if action_class.action_name == 'Finish':
                output = action_class(node, self.model_server, few_shot=self.few_shot, first=self.training and self.first_round,  direct_output=(node.value >= self.config.expand.bear_ratio and (self.training or mcts_inference)), **sampling_params)
            else:
                output = action_class(node, self.model_server, few_shot=self.few_shot, first=self.training and self.first_round, **sampling_params)
            # pdb.set_trace()
            
            for i in range(len(output)):
                observations[index[i]] = output[i].strip()
            # else:
            #     print(f"{action} is not supported in the current action space: {self.base_actions}")
            #     raise NotImplementedError

            # pdb.set_trace()
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

    def multisource_rag(self, node: MedMCTSNode, action_list: list[str], **sampling_params):
        source_list = getattr(getattr(self.config, "Medrag", {}), "source_list", ["MedCorp"])
        observations = []
        assert len(source_list) == len(action_list)
        for source in source_list:
            params = parse_action_params(action_list[0], self.config)
            params['corpus_name'] = source 
            action_class = Medrag(**params)
            output = action_class(node, self.model_server, few_shot=self.few_shot, **sampling_params)
            observations.append(output[0].strip())
        return observations
    
    def derive_answer(self, node: MedMCTSNode, **sampling_params):
        action_class = Finish()
        sampling_params['n'] = 1
        step = action_class(node, self.model_server, few_shot=0, first=self.training and self.first_round, direct_output=True, **sampling_params)

        step = step[0]

        new_node = MedMCTSNode(node.problem, step.strip(), 0, parent=node, type='Finish', ground_truth=node.ground_truth)
        node.add_child(new_node)
    
    def tot_bfs(self, beam_size=8, finish_candidates=16, value_op='identity', **sampling_params):
        # conduct beam search, every layer maintains the nodes with highest `beam_size` value 
        # maintain a list to store the finish nodes
        # this list should make the node with the lowest value to be popped once a new Finish node is generated and this new node has higher value than the top node
        # this list should be a priority queue, and the priority is the value of the node
        try:
            value_op = eval(value_op)
            value_op([1,2,3])
        except:
            value_op = lambda x: x[-1]
        candidate_nodes = [self.root]
        
        finish_nodes = PriorityQueue()
        # left_nodes = PriorityQueue()
        could_derive_answer_layer = 4
        while candidate_nodes:
            candidate_children_nodes = []
            for node in candidate_nodes:
                if node.type == 'Finish':
                    if finish_nodes.full():
                        finish_nodes.get()
                    finish_nodes.put((value_op(node.value_trajectory), node))
                    continue
                try:
                    if node.depth >= 9 or (node.value >= 0.9 and node.depth >= could_derive_answer_layer):
                        # a very deep node, directly expand Finish node
                        self.derive_answer(node, **sampling_params)
                    else:
                        if node.depth == 0:
                            max_children = finish_candidates 
                        else:
                            max_children = beam_size
                        # whether_expand_finish = (not node.depth >= could_derive_answer_layer)
                        self.tot_expand_node(node, max_children=max_children, whether_expand_finish=False, **sampling_params)

                except AttributeError as e:
                    logger.error(f"Error in expanding node: {node}")
                    continue
                candidate_children_nodes += node.children
                for child in node.children:
                    value, _ = self.rollout(child, self.value_func_simulation)
                    child.value = value
                    child.value_trajectory = node.value_trajectory + [value]
            # now for each node in candidate_nodes, we have expanded its children and obtain values
            # sort the candidate_children_nodes based on its value
            candidate_children_nodes.sort(reverse=True, key=lambda x: value_op(x.value_trajectory))
            # candidate_nodes = candidate_children_nodes[:beam_size]
            candidate_nodes = candidate_children_nodes[:finish_candidates]

        result = [x[1] for x in finish_nodes.queue[-finish_candidates:]]
        
        # while not finish_nodes.empty():
        #     result.append(finish_nodes.get()[1])
        return result
    
    
    def tot_dfs(self, beam_size=3, finish_candidates=16, stop_finish_nodes=32, value_op='identity', **sampling_params):  
        # conduct dfs on the tree; maintain a list to store the nodes that can be expanded
        # once expand a node, select its child with the highest value
        # if the child is a Finish node, add it to the finish_nodes, and back traverse to its father nodes
        try:
            value_op = eval(value_op)
            value_op([1,2,3])
        except:
            value_op = lambda x: x[-1]
        # candidate_nodes = [self.root]
        finish_nodes = PriorityQueue()
        could_derive_answer_layer = 4
        node = self.root
        while node and finish_nodes.qsize() < stop_finish_nodes:
            node.visits = 1
            if (node.value < 0.3 and (not finish_nodes.empty())) and node.depth > 1:
                # a pre-defined threshold to filter out low quality nodes
                # allow it has low value at first layer, but strict for deeper layer
                node = node.parent
                continue
            if node.type == 'Finish':
                if finish_nodes.full():
                    finish_nodes.get()
                finish_nodes.put((value_op(node.value_trajectory), node))
                node = node.parent
                continue 
            if not node.children:

                if node.depth >= 9 or (node.value >= 0.9 and node.depth >= could_derive_answer_layer):
                    # a very deep node, directly expand Finish node
                    self.derive_answer(node=node, **sampling_params)
                else:
                    # whether_expand_finish = (not node.depth >= could_derive_answer_layer)
                    
                    self.tot_expand_node(node, max_children=beam_size, whether_expand_finish=False, **sampling_params)
                for child in node.children:
                    value, _ = self.rollout(child, self.value_func_simulation)
                    child.value = value 
                    child.value_trajectory = node.value_trajectory + [value]
                    child.parent = node 

            # sort the children based on its value
            # candidate_children_nodes = sorted(explore_node.children, reverse=True, key=lambda x: value_op(x.value_trajectory))
            candidates = [k for k in node.children if k.visits == 0 and (k.value > 0.3 or k.depth <= 1 or finish_nodes.qsize() < finish_candidates)]
            if not candidates:                
                node = node.parent 
            else:
                node = max(candidates, key=lambda x: value_op(x.value_trajectory))
                

        
        result = [x[1] for x in finish_nodes.queue]
        # while not finish_nodes.empty() and len(results) < finish_candidates:
        #     result.append(finish_nodes.get()[1])
        return result
        
    
    def tot_inference(self, expand_way='bfs', beam_size=4, finish_candidates=16, infer_rule='prm-gmean-vote-sum', **sampling_params):
        if isinstance(self.autostep, int):
            sampling_params['max_tokens'] = self.autostep
        elif isinstance(self.autostep, str):
            sampling_params['stop'] = [f"Step {i}:" for i in range(1, 100)]
        
        if expand_way == 'bfs':
            leaves = self.tot_bfs(beam_size, finish_candidates, value_op='identity', **sampling_params)
        elif expand_way == 'dfs':
            leaves = self.tot_dfs(beam_size, finish_candidates, value_op='identity', **sampling_params)
        else:
            raise NotImplementedError
        only_answer_outputs = [extract_template(leaf.reasoning_step, 'answer') for leaf in leaves]
        values = [[0] + leaf.value_trajectory for leaf in leaves]
        max_answer, weighted_values = compute_weighted_values(only_answer_outputs, values, infer_rule)
        
        max_weighted_value = weighted_values[max_answer]
        tie_answers = [ans for ans, val in weighted_values.items() if abs(val - max_weighted_value) < 1e-6]
        value_op = infer_rule.split("-")[1] if infer_rule.startswith('prm') else 'identity'
        try:
            value_op = eval(value_op)
        except:
            value_op = lambda x: x[-1]
        if len(tie_answers) == 1:
            # 没有平局，直接选择 max_answer 对应的节点
            # best_node = 
            best_node =  max((node for node in leaves if extract_template(node.reasoning_step, 'answer') == max_answer), key=lambda n: n.value)
        else:
            best_node = None
            best_value = float('-inf')
            for leaf in leaves:
                
                if extract_template(leaf.reasoning_step, 'answer') in tie_answers and value_op(leaf.value_trajectory) > best_value:
                    best_value = value_op(leaf.value_trajectory)
                    best_node = leaf

        return leaves, best_node
    
    def print_tree(self, node):
        if node.children:
            children_str = ", ".join([f"Node ({child.trace})" for child in node.children])
            print(f"[Trace: {node.trace}, Value: {node.value}, Visits: {node.visits}] -> [{children_str}]")
        else:
            print(f"[Trace: {node.trace}, Value: {node.value}, Visits: {node.visits}]")
        for child in node.children:
            self.print_tree(child)
            

def print_node(node: MedMCTSNode):
    print(node)
    print("*" * 80)
    
    for child in node.children:
        print_node(child)  

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
    from Evol_Instruct.models.modeling_value_llama import ValueModel
    torch.set_default_device("cuda")
    # model_path = '/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct'
    model_path = '/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3.1-8B-Instruct-ysl'
    # model_path = '/mnt/hwfile/medai/LLMModels/Model/Mistral-7B-Instruct-v0.3'
    # model_path = '/mnt/hwfile/medai/LLMModels/Model/Qwen2.5-7B-Instruct'
    # model_path = '/mnt/hwfile/medai/LLMModels/Model/Phi-3.5-mini-instruct'
    # model_path = '/mnt/hwfile/medai/LLMModels/Model/gemma-2-9b-it'
    # model_path = ''
    # model_path = '/mnt/petrelfs/jiangshuyang.p//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-SFT-full-ITER1'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = set_tokenizer(tokenizer)
    # data = client.read("s3://syj_test/datasets/medical_train/pubhealth.json")
    data = client.read("/mnt/petrelfs/jiangshuyang.p/datasets/medical_test/MedQA_cot.json")
    # data = client.read("/mnt/petrelfs/jiangshuyang.p/datasets/medical_train/medqa_train.json")
    # for i in range(15):
    # model_base = LlamaForValueFunction.from_pretrained(
    #     "/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct",
    #     num_labels=1
    # )
    # model_base = AutoModelForSequenceClassification.from_pretrained(
    #     # "/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct",
    #     model_path,
    #     # "/mnt/petrelfs/jiangshuyang.p//checkpoints/llama38b_mcts_vllm_medqa_train_all_search_1000/sft_1-llama3-8b-r16a32-1epoch-VALUE-full-ITER1/checkpoint-100",
    #     num_labels=1
    # )
    sft_model_path = '/mnt/petrelfs/jiangshuyang.p//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-SFT-ITER1'
    # reward_model_path = '/mnt/petrelfs/jiangshuyang.p//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-VALUE-prm_train5_r64-ITER1'
    reward_model_path = '/mnt/petrelfs/jiangshuyang.p//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-VALUE-prm_trainall_r64_softtrain_basepolicy-ITER1'
    value_model = ValueModel(model_path, [sft_model_path,reward_model_path], 'prm')
    
    # # # sft_model = PeftModel.from_pretrained(model_base, sft_model_path).merge_and_unload()
    # value_model = PeftModel.from_pretrained(model_base, reward_model_path).merge_and_unload().to(torch.float16)
    # print(value_model.config)
    # value_model = PeftModel.from_pretrained(model_base, "/mnt/petrelfs/jiangshuyang.p//checkpoints/llama38b_mcts_vllm_medqa_train_all_search_1000/sft_1-llama3-8b-r16a32-1epoch-VALUE_new2-ITER1")
    # value_model = value_model.merge_and_unload().to(torch.float16)
    # value_model = model_base
    # value_model = None
    # server = VLLMServer(url="http://10.140.1.163:10003", model=model_path, tokenizer=tokenizer, offline=True)
    # server = VLLMServer(url="http://10.140.1.163:10002", model=model_path, tokenizer=tokenizer, offline=True, lora_path=None, gpu_memory_usage=0.45, max_model_len=16384)
    server = VLLMServer(url="http://10.140.1.163:10002", model=model_path, tokenizer=tokenizer, offline=True, gpu_memory_usage=0.5, lora_path=sft_model_path, max_model_len=16384)
    # exit(0)
    while 1:
        i = int(input("Input the test number ID: "))
        # i = 3
        item = data[i]
        logger.info(item)
        # logger.info(f"Correct answer: {item['answer_idx']}")
        if 'answer_idx' not in item:
            item['answer_idx'] = item['eval']['answer']
        # # print(node)
        config = client.read(path="Evol_Instruct/config/trial5.json")
        config = MCTSConfig(config)
        # config.terminate.max_nodes = 120
        # config.expand.unvisited_ucb = 2

        # config.max_children=3
        config.terminate.least_leaves = 16
        mcts_cls = tree_registry[config.mcts_cls]
        # config.expand.max_children = 16
        tree: MCTS = mcts_cls(item, model_server=server, config=config, value_function=value_model, training=False)
        tokenizer = tree.tokenizer 
        start = time.time()
        # leaves, best_node = tree.tot_inference(expand_way='dfs', beam_size=3)
        # leaves, best_node = tree.tot_inference(expand_way='bfs', beam_size=6)
        root = tree.run(temperature=1)
        # node, node_steps = tree.inference("vote-sc")
        # exit(0)
        # correct_leaves = tree.obtain_correct_leaves()
        # incorrect_leaves = tree.obtain_incorrect_leaves()
        # repeat_try = config.repeat_try
        # while not correct_leaves and repeat_try > 0:
        #     tree = MCTS(item, model_server=server, config=config, value_function=value_model, training=False)
        #     root = tree.run()
        #     correct_leaves = tree.obtain_correct_leaves()
        #     repeat_try -= 1
        end = time.time()
        # print(node.obtain_reasoning_steps()[0], flush=True)
        # pdb.set_trace()
        client.write("", "debug.log", mode='w')
        fp = open("debug.log", 'a')
        leaves = tree.obtain_leaves(root)
        leaves = [leaf for leaf in leaves if leaf.type == 'Finish']
        weighted_values = defaultdict(float)
        count_values = defaultdict(int)
        for leaf in leaves:
            answer = extract_template(leaf.obtain_reasoning_steps()[0], 'answer')
            if answer is not None:
                weighted_values[answer] += leaf.value 
                count_values[answer] += 1
        print(f"Weighted values: {weighted_values}\nCount values: {count_values}", file=fp, flush=True)
        for leaf in leaves:
            print(leaf.obtain_reasoning_steps()[0], leaf.value_chain(), file=fp, flush=True)
            print("*" * 100, file=fp, flush=True)
        
        node, node_steps = tree.inference('prm-prod-vote-sum')
        print("PRM GMean Vote Sum:", node_steps, node.value_chain(), file=fp, flush=True)
        
        node, node_steps = tree.inference('prm-prod-vote-mean')
        print("PRM GMean Vote Mean:", node_steps, node.value_chain(), file=fp, flush=True)
        # node, node_steps = tree.inference('prm-prod-vote-sum')
        # print("PRM Production Vote Sum:", node_steps, node.value_chain(), file=fp, flush=True)
        
        # node, node_steps = tree.inference("vote-mean")
        # print("Vote Mean:", node_steps, node.value_chain(), file=fp, flush=True)
        
        # node, node_steps = tree.inference("vote-sum")
        # print("Vote-Sum:", node_steps, node.value_chain(), file=fp, flush=True)
        
        # node, node_steps = tree.inference("level_max")
        # print("level-max:", node_steps, node.value_chain(), file=fp, flush=True)
        
        print("*" * 100, file=fp, flush=True)
        
        logger.info(f"Time cost: {end - start}")
    