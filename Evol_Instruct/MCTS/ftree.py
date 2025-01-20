from collections import defaultdict
import time
from typing import Counter

import torch

from Evol_Instruct.MCTS.tree import MCTS, MCTSConfig, MedMCTSNode
from Evol_Instruct.MCTS.tree_register import register_tree
from Evol_Instruct.MCTS.utils import extract_template, parse_action_params
from Evol_Instruct import logger, client
from Evol_Instruct.evaluation.generate_utils import set_tokenizer

from Evol_Instruct.models.vllm_support import VLLMServer
from Evol_Instruct.MCTS.tree_register import tree_registry
from Evol_Instruct.actions.base_action import Finish, Reflect, Think, Reason, Refine

@register_tree("FMCTS")
class FMCTS(MCTS):
    # def __init__(*args, **args)
    def lookahead_one_step(self, node: MedMCTSNode, **sampling_params):
        action_list = ['Reason'] * self.config.expand.max_children
        next_observations = self.step_observation(node, action_list, **sampling_params)
        max_nodes_value = -1
        max_node = None
        for each in next_observations:
            # node.add_child()
            each_node = MedMCTSNode(node.problem, each.strip(), len(node.children), parent=node, type='Reason', ground_truth=node.ground_truth)
            # node.add_child(each_node)
            if extract_template(each_node.reasoning_step, 'answer') is not None:
                value, simu_value = each_node.eval_node(self.value_func_simulation, self.training)
                if self.training:
                    each_value = simu_value
                else:
                    each_value = value 
            else:
                each_value, _ = self.value_func_simulation(each_node)
            if each_value > max_nodes_value:
                max_nodes_value = each_value 
                max_node = each_node
        return max_node, max_nodes_value
    
    def future_rollout(self, node: MedMCTSNode, **sampling_params):
        # action = Reason()
        # lookahead two steps
        cur_value, _ = self.value_func_simulation(node)
        value_list = [cur_value]
        max_node, max_nodes_value = self.lookahead_one_step(node, **sampling_params)
        node.add_child(max_node)
        value_list.append(max_nodes_value)
        if extract_template(max_node.reasoning_step, 'answer') is not None:
            # max_node.type = 'Finish'
            true_value = 0.5 * (value_list[0] + max(value_list[1:]))
            node.delete_child()
        else:
            # next_observations = self.step_observation(max_node, action_list, **sampling_params)
            # max_nodes_value = -1
            new_max_node, new_max_nodes_value = self.lookahead_one_step(max_node, **sampling_params)
            # max_node.add_child(new_max_node)
            value_list.append(new_max_nodes_value)
            true_value = 0.5 * (value_list[0] + max(value_list[1:]))
            node.delete_child()
        return true_value, None
            # max_node.eval_node()
        # pass
    
    def step_observation(self, node: MedMCTSNode, action_list: list[str], **sampling_params):
        observations = [None for _ in range(len(action_list))]
        # obtain action_list count
        action_count = Counter(action_list)
        action_index = defaultdict(list)
        for i, action in enumerate(action_list):
            action_index[action].append(i)
        for action, index in action_index.items():
            count = action_count[action]
            sampling_params['n'] = count
            if not action in self.base_actions:
                params = parse_action_params(action, self.config)
            else:
                params = {}

            action_class = eval(action, globals())(**params)

            output = action_class(node, self.model_server, few_shot=self.few_shot, 
                                  first=self.model_server.lora_request is None, **sampling_params)
            # pdb.set_trace()
            
            for i in range(len(output)):
                observations[index[i]] = output[i].strip()
            # else:
            #     print(f"{action} is not supported in the current action space: {self.base_actions}")
            #     raise NotImplementedError

            # pdb.set_trace()
        return observations
    
    
    def rollout(self, node: MedMCTSNode, simu_func, simulation=20, **sampling_params):
        if node.type == 'Finish':
           
            # value = node.eval_node()
            if self.value_function is None:
                value = node.eval_node(None)
            else:
                value = self.value_func_simulation(node)
                # value = simu_func(node)
            if "Reflect" in [action.action_name for action in self.actions]:
                if value[0] < self.low_gate:
                    node.type = 'Reason'
            return value 
        if self.value_function is None:
            return simu_func(node, simulation)
        else:
            return self.future_rollout(node, **sampling_params)
    
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
        
        # logger.debug(f"Selected Node in Iteration {iter_time}: {node}")
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
        elif node.visits == 0 and node.depth > 0:
            if self.value_function is not None:
                # inference mode
                value, simulation_score = self.rollout(node, self.value_func_simulation, **sampling_params)
            else:
                # pdb.set_trace()
                value, simulation_score = self.rollout(node, self.direct_infer_simulation, simulation=self.config.simulation, **sampling_params)
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
    
    def post_process(self, node: MedMCTSNode, value_function=None, **sampling_params):
        # once terminated, check if all leaf nodes is a Finish node, if not,
        # expand one child to obtain the Finish node
        if node.is_completed:
            return
        leaves = self.obtain_leaves(self.root)
        if len(leaves) == 0:
            finish_uncompleted = True
        else:
            finish_uncompleted = getattr(self.config, 'finish_uncompleted', True)
        
        for leaf in leaves:
            if leaf.type != 'Finish' and finish_uncompleted:
                action = 'Finish'
                # cur_prompt = mcts_prompts['action_template'].format(
                #     problem=node.problem,
                #     steps=node.obtain_reasoning_steps()[0],
                #     action=mcts_prompts[action]
                # )
                sampling_params['n'] = 1
                action_class = Finish()
                step = action_class(node, self.model_server, few_shot=self.few_shot, first=self.value_function is None, **sampling_params)
                # step = self.step_llm_action(node, action, **sampling_params)
                # reasoning_steps = self.model_server(prompt=[cur_prompt], **sampling_params)
                step = step[0]
                
                # step = reasoning_steps[0][0]
                new_node = MedMCTSNode(node.problem, step, 0, parent=leaf, type=action, ground_truth=node.ground_truth)
                leaf.add_child(new_node)
                value, simu_value = new_node.eval_node(value_function, training=self.training)
                self.back_propagate(new_node, value=float(value), simulation_score=simu_value)
            # elif leaf.type != 'Finish' and not finish_uncompleted:
            #     # delete this node if this node has never been visited
            #     temp = leaf 
            #     while temp.visits == 0 and temp.type != 'Finish':
            #         # temp.parent.delete_child(temp.index)
            #         temp.parent.children[temp.index] = None
            #         temp = temp.parent
                # if leaf.visits == 0:
                #     leaf.parent.delete_child(leaf.index)
                    
            elif leaf.type == 'Finish':
                if leaf.visits == 0:
                    value, simu_value = leaf.eval_node(value_function, training=self.training)
                    self.back_propagate(leaf, value=float(value), simulation_score=simu_value)
        return


