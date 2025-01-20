from Evol_Instruct.MCTS.lstree import LSMCTS
from Evol_Instruct.MCTS.mtree import MMCTS
from Evol_Instruct.MCTS.tree import MCTS, MCTSConfig
from Evol_Instruct.MCTS.utils import extract_template

from Evol_Instruct.models.vllm_support import VLLMServer
from Evol_Instruct.solver.base_solver import Solver
from Evol_Instruct.utils.utils import AlpacaTaskItem
import json
import time
from Evol_Instruct.evaluation.generate_utils import task_specific_prompt_mapping, CustomDataset, infer_answer, set_tokenizer
from Evol_Instruct.MCTS.tree_register import tree_registry

class MCTSSolver(Solver):
    def __init__(self, *args, mcts_config, value_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = mcts_config
        self.value_model = value_model
        self.infer_rule = kwargs.pop("infer_rule", "max")
        self.autostep = getattr(self.config.expand, 'autostep', 'step')
        if isinstance(self.autostep, str):
            if getattr(self.config, 'manual', False):
                self.mcts_cls = MMCTS
            else:
                # self.mcts_cls = MCTS
                self.mcts_cls = tree_registry.get(getattr(self.config, 'mcts_cls', 'MCTS'), MCTS)
        elif isinstance(self.autostep, int):
            self.mcts_cls = LSMCTS
    
    def infer_with_level_max(self, tree):
        # text = 
        tree = json.loads(tree)
        trace = [tree['step']]
        temp = tree
        while temp['children'] != []:
            value = -1
            next_level = None
            for child in temp['children']:
                if child['value'] > value:
                    value = child['value']
                    next_level = child 
            trace.append(next_level['step'])
            temp = next_level 
        return "\n\n".join([f"Step {i}: {step}" for i, step in enumerate(trace)])
        
    
    def generate_response(self, items: list[AlpacaTaskItem], **kwargs):
        for item in items:
            tree = self.mcts_cls(item.prompt, self.server, self.config,
                         self.value_model, training=False)
            if self.infer_rule.startswith('tot'):
                expand_way = kwargs.pop("expand_way", "bfs")
                beam_size = self.config.expand.max_children
                finish_candidates = kwargs.pop("finish_candidates", 16)
                infer_rule = self.infer_rule[4:]
                finish_nodes, node = tree.tot_inference(expand_way, beam_size, finish_candidates, infer_rule)
                output = node.obtain_reasoning_steps()[0]

                # node, output = tree.inference(self.infer_rule, **kwargs)
            else:
                value_model_type = kwargs.pop("value_model_type", "prm")
                expand_way = kwargs.pop("expand_way", "bfs")
                root = tree.run(**kwargs)
                # item.tree = root.output_tree()
                # item.tree = str(root)
                node, output = tree.inference(self.infer_rule)
                if extract_template(output, 'answer') is None:
                    answer_outputs = self.infer_answer([item.prompt], [output], self.choices_word)
                    output = answer_outputs[0]
                finish_nodes = [node for node in tree.obtain_leaves(tree.root) if node.type == 'Finish']
            
            item.text = output 
            item.trajectory = node.get_trajectory()
                
            finish_answer = [node.reasoning_step for node in finish_nodes]
            # item.all_answer = [(a, v.value) for a, v in zip(finish_answer, finish_nodes)]
            finish_value_chain = [node.value_trajectory for node in finish_nodes]   
            
            # for node in finish_nodes:
            #     cur_value_chain = []
            #     temp = node
            #     while temp.parent:
            #         cur_value_chain.append(temp.value)
            #         temp = temp.parent
            #     cur_value_chain = cur_value_chain[::-1]
            #     finish_value_chain.append(cur_value_chain)
                
            item.all_answer = [[a, value_chain] for a, value_chain in zip(finish_answer, finish_value_chain)]
        
        return items


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from Evol_Instruct import client
    from Evol_Instruct.evaluation.generate_utils import CustomDataset
    from Evol_Instruct.models.modeling_value_llama import ValueModel
    data = client.read("s3://syj_test/datasets/medical_test/MedQA_cot.json")
    model_base='/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3.1-8B-Instruct-ysl'
    lora_path='/mnt/petrelfs/jiangshuyang.p//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-SFT-ITER1'
    reward_model_path = '/mnt/petrelfs/jiangshuyang.p//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-VALUE-prm_trainall3_r64-ITER1'
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer = set_tokenizer(tokenizer)
    server = VLLMServer(url='http://10.140.1.163:10002', 
                        model=model_base,
                        lora_path=lora_path,
                        tokenizer = tokenizer,
                        offline=True, gpu_memory_usage=0.45)
    value_model = ValueModel(model_base, reward_model_path, 'prm')
    config = client.read("/mnt/petrelfs/jiangshuyang.p/repo/new/WizardLM/Evol_Instruct/config/trial5_2.json")
    mcts_config = MCTSConfig(config)
    dataset = CustomDataset(data, 1, "", tokenizer=tokenizer, )
    solver = MCTSSolver(server, open('temp.jsonl', 'w'), ['A', 'B', 'C', 'D', 'E'], value_model=value_model, infer_rule='prm-gmean-vote-sum', mcts_config=mcts_config, cot_prompt="\nThe answer is ")
    
    
    output = solver.generate_response(dataset[1], max_tokens=3072, temperature=1, n=1, top_p=0.95, value_model_type='prm', expand_way='bfs')
    solver.save_response(output)