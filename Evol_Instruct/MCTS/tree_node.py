from typing import Optional
from Evol_Instruct.MCTS.utils import extract_template
from Evol_Instruct.prompts.prompt_template import mcts_prompts
import numpy as np
import json 


class MCTSNode:
    def __init__(self, problem: str, reasoning_step: str, index: int, parent=None, ground_truth=None, type='Think', prompt=None, type_param=None):
        self.index = index  # 推理的当前状态
        self.reasoning_step = reasoning_step
        self.problem = problem
        self.prompt = prompt
        self.parent: Optional[MCTSNode] = parent  # 父节点
        self.children: list[MCTSNode] = []  # 子节点
        self.visits = 0  # 访问次数
        self.value = 0  # 价值v
        self.value_trajectory = []
        self.simulation_score = None
        self.is_completed = False  # 该节点是否是完成状态
        self.depth = parent.depth + 1 if parent else 0  # 节点深度
        self.ground_truth = ground_truth
        # self.index = 0 if self.parent is None else self.parent.children.index(self)
        self.type = type  # 节点类型
        self.type_param = type_param  # 节点参数
        self.type_output = reasoning_step
        self.trace = [0]
        self.correct = False
        # self.incorrect = None
        if self.parent is not None:
            self.trace = self.parent.trace + [self.index]
        else:
            self.trace = [0]

    def add_child(self, child):
        # child = MCTSNode(child_state, parent=self)
        self.children.append(child)
        return child


    def __lt__(self, other):
        return self.value < other.value 

    # def __
    
    def delete_child(self, child_index=-1):
        self.children.pop(child_index)
        return 
    
    def obtain_reasoning_steps(self):
        trace_nodes = []
        node = self
        while node:
            trace_nodes.append(node)
            node = node.parent
        trace_nodes = trace_nodes[::-1]
        output_string = ""
        for i, node in enumerate(trace_nodes):
            output_string += f"Step {i}: {node.type_output}\n\n"
            
        output_string = output_string.strip("\n")
        if output_string == "":
            return "None"
        return output_string, len(trace_nodes)
        
        
    def __str__(self):
        format_str = {
            "step": self.reasoning_step,
            "value": self.value,
            "parent": str(self.parent.reasoning_step) if self.parent is not None else None,
            "type": self.type,
            "children": [child.__str__() for child in self.children]
        }
        return json.dumps(format_str)
       
    
    def __repr__(self) -> str:

        return f"Node: {self.reasoning_step}\nType: {self.type}\nVisits: {self.visits}\nDepth: {self.depth}\nTrace: {self.trace}\nValue Trajectory: {self.value_trajectory}"
    
    

class MedMCTSNode(MCTSNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_traces = self.obtain_reasoning_steps()[0]
    
    def __lt__(self, other):
        return self.value < other.value 
    
    def is_correct(self, pred: str, answer: str):

        if len(answer) == 1:
            # indicate that this is a multiple-choice problem
            # remove any quote strings from the pred
            clean_pred = pred.replace("'", "").replace('"', '').replace("*", "")
            return answer in clean_pred[0]
        elif len(answer) != 1:
            # indicate that this is the open-ended problem
            clean_answer = answer.lower()
            clean_pred = pred.lower()
            return clean_answer in clean_pred

    def eval_node(self, value_function=None, training=False):

        self.is_completed = True
        
        if value_function is None:
            answer = extract_template(self.reasoning_step, "answer")
            if answer is None:
                value = 0
            else:
                value = float(self.is_correct(answer, self.ground_truth))
            simu_value = value
        else:
            # maybe the second iteration's training or inference
            simu_value = None
            if training:
                answer = extract_template(self.reasoning_step, "answer")
                if answer is None:
                    simu_value = 0
                else:
                    simu_value = float(self.is_correct(answer, self.ground_truth))
            value, _ = value_function(self, training=False)
        return value, simu_value
    

    def build_simu_prompt(self):
        simulate_prompt = mcts_prompts['simu_template'].format(
            problem=self.problem,
            steps=self.obtain_reasoning_steps()[0],
            Simulate=mcts_prompts['Simulate']
        )
        return simulate_prompt
    
    

    
    def calculate_diversity(self):
        if hasattr(self, "diversity"):
            return self.diversity
        all_traces = set()
        def traverse(node):
            if node:
                all_traces.add(tuple(node.trace))
                for child in node.children:
                    traverse(child)
        traverse(self)
        self.diversity = len(all_traces)
        return self.diversity 
    
    def calculate_uncertainty(self):
        if hasattr(self, "uncertainty"):
            return self.uncertainty
        visits = self.visits
        uncertainty = 0
        
        if visits > 0:
            uncertainty = 1 / visits 
        
        if hasattr(self, "simulations") and len(self.simulations) > 1:
            value_variance = np.var(self.simulations)
            uncertainty += value_variance
        
        self.uncertainty = uncertainty
        return uncertainty
    
    def calculate_score(self, alpha=0.5):
        diversity = self.calculate_diversity()
        uncertainty = self.calculate_uncertainty()
        score = alpha * diversity + (1 - alpha) * uncertainty
        return score
    
    def select_representative_nodes(self):
        node_scores = []
        def traverse(node):
            if node is not None:
                if node.children and node.parent is not None:
                    # not leaf node and not root
                    score = self.calculate_score()
                    node_scores.append((node, score))
                for child in node.children:
                    traverse(child)
        
        traverse(self)
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_n = len(node_scores) // 2
        top_n_nodes = [x[0] for x in node_scores[:top_n]]
        return top_n_nodes
    
    def select_all_intermediate_nodes(self):
        nodes = []
        def traverse(node):
            if node is not None:
                if node.children and node.parent is not None:
                    # not leaf node and not root
                    nodes.append(node)
                elif (not node.children) and node.type != 'Finish' and node.visits > 0:
                    nodes.append(node)
                for child in node.children:
                    traverse(child)
        
        traverse(self)
        return nodes
    
    def get_trajectory(self):
        assert self.children == [], "Only leaf node can get trajectory"
        node = self
        trajectory = [{"step": node.reasoning_step, "value": node.value}]
        while node.parent:
            node = node.parent 
            trajectory.append({"step": node.reasoning_step, "value": node.value})
        return trajectory[::-1]
    
    def __eq__(self, other):
        if isinstance(other, MedMCTSNode):
            return self.reasoning_traces == other.reasoning_traces
        return False 
    
    def __hash__(self):
        return hash(self.reasoning_traces)

    def value_chain(self):
        # value_format = ""
        chain = []
        temp = self
        while temp:
            chain.append(json.dumps({f"Node ({temp.type})": temp.value, "index": temp.index}))
            temp = temp.parent
        chain = chain[::-1]
        chain_str = " -> ".join(chain)
        return chain_str
    
    def reasoning_chain(self):
        chain = []
        temp = self
        while temp:
            chain.append([temp.reasoning_step, temp.value, temp.simulation_score])
            temp = temp.parent
        chain = chain[::-1]
        return chain
