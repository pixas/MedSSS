import math
from Evol_Instruct.MCTS.tree import MCTS, MCTSConfig
from Evol_Instruct.MCTS.tree_node import MedMCTSNode
from Evol_Instruct.MCTS.tree_register import register_tree
from Evol_Instruct.MCTS.utils import extract_template
from Evol_Instruct import logger
import time 
from Evol_Instruct import client
import torch 
from Evol_Instruct.evaluation.generate_utils import set_tokenizer

from Evol_Instruct.models.vllm_support import VLLMServer

@register_tree("MMCTS")
class MMCTS(MCTS):
    def normal_expand_node(self, node: MedMCTSNode, max_children=3, **sampling_params):
        sampling_params['n'] = max_children
        # cur_action = 'Think'
        # think_action = Think(self.actions)
        # step_count = len(node.trace)
        # if node.depth != 0:
        #     if node.type == 'Medrag':
        #         next_actions = ['Reason'] * max_children
        #     else:
                # next_actions = think_action(node, self.model_server, **sampling_params)
                
        # pdb.set_trace()
        # else:
        #     next_actions = ['Reason'] * max_children
        # if not next_actions:
        next_actions = ['Reason'] * max_children
        return next_actions
    
    def expand_node(self, node: MedMCTSNode, max_children=3, bear_ratio=0.9,
                    low_gate=0.3, **sampling_params):
        if node.is_completed:
            return 
        if node.children != []:
            return 

        if node.simulation_score is not None and node.simulation_score >= bear_ratio:
            # training
            max_children = 1
            next_actions = ['Finish']
        elif node.simulation_score is None and node.value >= bear_ratio:
        # if node.value >= bear_ratio:
            # inference
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
                        # sampling_params['n'] = max_children
                        # max_children = 1
                        action = 'Reflect'
                        
                        # max_children = 1
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
        if "Medrag" in next_actions:
            observations = self.multisource_rag(node, next_actions, **sampling_params)
        else:
            observations = self.step_observation(node, next_actions, **sampling_params)
        


        # pdb.set_trace()       
        for i in range(len(next_actions)):
            step = observations[i].strip("\n")
            step = step.replace("<steps>", "") if step.startswith("<steps>") else step
            
            # step = step.replace("<steps>", "").strip("\n")
            if next_actions[i] == 'Finish':
                self.process_answer_nodes(node, step, idx=i)
            else:
                if extract_template(step.strip(), 'answer') is not None:
                    next_actions[i] = 'Finish'
                new_node = MedMCTSNode(node.problem, step.strip(), i, parent=node, type=next_actions[i], ground_truth=node.ground_truth)
                node.add_child(new_node)
    
    def select_child(self, node, constant=2):
        max_ucb = -1
        return_node = None
        constant_change = getattr(self.config.expand, 'constant_change', 'constant')
        # if constant_change != 'constant':
        #     constant = eval(constant_change)
        for child in node.children:
            if constant_change != 'constant':
                cur_constant = eval(constant_change)
            else:
                cur_constant = constant
            if child.is_completed:
                # all its children are explored and hence it is completed
                # choose not to explore this trace
                if cur_constant > 0:
                    ucb = -1
                
                # if self.training:
                #     ucb = -1
                else:
                    ucb = child.value / child.visits + cur_constant * math.sqrt(math.log(node.visits) / child.visits)
            elif child.visits == 0:
                ucb = getattr(self.config.expand, 'unvisited_ucb', math.inf)
            else:
                ucb = child.value / child.visits + cur_constant * math.sqrt(math.log(node.visits) / child.visits)
            if ucb > max_ucb:
                if max_ucb == getattr(self.config.expand, 'unvisited_ucb', math.inf):
                    logger.info("Explore already visited nodes")
                max_ucb = ucb
                return_node = child

        return return_node

