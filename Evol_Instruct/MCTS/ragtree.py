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

from Evol_Instruct.actions.base_action import Finish, Reflect, Think, Reason, Refine
from Evol_Instruct.MCTS.tree_register import tree_registry
from collections import defaultdict

@register_tree("RAGMCTS")
class RAGMCTS(MCTS):
    def normal_expand_node(self, node: MedMCTSNode, max_children=3, **sampling_params):
        # sampling_params['n'] = max_children
        # # cur_action = 'Think'
        # think_action = self.think_action
        # # step_count = len(node.trace)
        # if node.depth == 1:
        #     # if node.type == 'Medrag':
        #     #     next_actions = ['RAGReason'] * max_children
        #     # else:
        #     #     next_actions = think_action(node, self.model_server, **sampling_params)
        #     next_actions = ['RAGReason'] * max_children
        # # pdb.set_trace()
        # elif node.depth == 0:
        #     next_actions = ['Reason'] * max_children
        # else:
        #     next_actions = think_action(node, self.model_server, **sampling_params)
        # if not next_actions:
        #     next_actions = ['RAGReason']
        # next_actions = ['RAGReason'] * max_children
        # next_actions = ['RAGReason'] * max_children
        if node.depth == 1:
            next_actions = ['RAGReason'] * max_children
        else:
            next_actions = ['Reason'] * max_children
        if node.depth > 3:
            next_actions = self.think_action(node, self.model_server, **sampling_params)
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
            

        if "" in next_actions:
            next_actions = ['Finish'] * len(next_actions)
        
        
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
                new_node = MedMCTSNode(node.problem, step.strip(), i, parent=node, type=next_actions[i], ground_truth=node.ground_truth,)
                node.add_child(new_node)
    
    def process_answer_nodes(self, node: MedMCTSNode, reasoning_step: str, idx: int):
        steps = reasoning_step.split("\n\nStep")
        steps = [step[4:] if i > 0 else step for i, step in enumerate(steps) ]
        if len(steps) > 1 and extract_template(steps[-1], "answer") is not None:

            new_node = MedMCTSNode(node.problem, steps[0].strip(), len(node.children), parent=node, type='Reason', ground_truth=node.ground_truth)
            
            node.add_child(new_node)
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
                node_type = 'RAGReason'
            new_node = MedMCTSNode(node.problem, steps[-1].strip(), len(node.children), parent=node, type=node_type, ground_truth=node.ground_truth)
            node.add_child(new_node)

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

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
    torch.set_default_device("cuda")
    model_path = '/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct'
    # model_path = '/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3.1-8B-Instruct'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = set_tokenizer(tokenizer)
    # data = client.read("s3://syj_test/datasets/medical_train/pubhealth.json")
    data = client.read("s3://syj_test/datasets/medical_test/MedQA_cot.json")
    # data = client.read("s3://syj_test/datasets/medical_train/mmed_en_train.json")
    # for i in range(15):
    # model_base = LlamaForValueFunction.from_pretrained(
    #     "/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct",
    #     num_labels=1
    # )
    model_base = AutoModelForSequenceClassification.from_pretrained(
        "/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct",
        num_labels=1
    )
    value_model = PeftModel.from_pretrained(model_base, "/mnt/petrelfs/jiangshuyang.p//checkpoints/llama38b_mcts_vllm_medqa_train_all_search_1000/sft_1-llama3-8b-r16a32-1epoch-VALUE_new2-ep2-ITER1")
    value_model = value_model.merge_and_unload().to(torch.float16)
    # value_model = model_base
    # value_model = None
    # server = VLLMServer(url="http://10.140.1.163:10002", model=model_path, tokenizer=tokenizer, offline=False)
    server = VLLMServer(url="http://10.140.1.163:10002", model=model_path, tokenizer=tokenizer, offline=True, lora_path='/mnt/petrelfs/jiangshuyang.p//checkpoints/llama38b_mcts_vllm_medqa_train_all_search_1000/sft_1-llama3-8b-r16a32-1epoch-SFT-ITER1', gpu_memory_usage=0.45, max_model_len=8192)
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
        config = client.read("Evol_Instruct/config/search.json")
        config = MCTSConfig(config)
        # config.finish_uncompleted = True
        # config.max_children=3
        
        # config.max_children = 6
        mcts_cls = config.mcts_cls
        mcts_cls = tree_registry.get(mcts_cls, "MCTS")
        tree = mcts_cls(item, model_server=server, config=config, value_function=value_model, training=False)
        tokenizer = tree.tokenizer 
        start = time.time()
        # node = tree.tot_inference()
        root = tree.run(temperature=0.7)
        answer_nodes = [leaf for leaf in tree.obtain_leaves(root) if leaf.type == 'Finish']
        if not answer_nodes:
            config.terminate.max_nodes += 30
            root = tree.run(temperature=0.7)
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
        weighted_values = defaultdict(float)
        count_values = defaultdict(int)
        for leaf in leaves:
            answer = extract_template(leaf.obtain_reasoning_steps()[0], 'answer')
            if answer is not None:
                weighted_values[answer] += leaf.value 
                count_values[answer] += 1
        # if weighted_values == {}:
        print(f"Weighted values: {weighted_values}\nCount values: {count_values}", file=fp, flush=True)
        # for leaf in leaves:
        #     print(leaf.obtain_reasoning_steps()[0], leaf.value_chain(), file=fp, flush=True)
        #     print("*" * 100, file=fp, flush=True)
        node, node_steps = tree.inference("vote-sc")
        print("Vote SC:", node_steps, node.value_chain(), file=fp, flush=True)
        
        node, node_steps = tree.inference("vote-mean")
        print("Vote Mean:", node_steps, node.value_chain(), file=fp, flush=True)
        
        node, node_steps = tree.inference("vote-sum")
        print("Vote-Sum:", node_steps, node.value_chain(), file=fp, flush=True)
        
        # node, node_steps = tree.inference("level_max")
        # print("level-max:", node_steps, node.value_chain(), file=fp, flush=True)
        
        print("*" * 100, file=fp, flush=True)
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
        # for leaf in leaves:
        #     # with open("debug.log")
        #     print(leaf.value_chain(), file=fp, flush=True)
        #     print(leaf.obtain_reasoning_steps()[0], file=fp, flush=True)
        #     print("*" * 80, file=fp, flush=True)

        # correct_leaves = tree.obtain_correct_leaves()
        # for leaf in correct_leaves:
        #     print(leaf.obtain_reasoning_steps()[0])
        #     print("*" * 80)
        logger.info(f"Time cost: {end - start}")
        # leaf, reasoning_step = tree.inference()
        # print("Max inference", reasoning_step, leaf.value_chain(), file=fp)
        # leaf, reasoning_step = tree.inference('vote')
        # print("Vote inference", reasoning_step, leaf.value_chain(), file=fp)
        # with open("datas/tree2.pkl", 'wb') as file:
        #     pickle.dump(root, file)
        # with open("datas/tree2.pkl", 'rb') as file:
        #     node = pickle.load(file)
        # print_node(root)
        # correct_path = root.obtain_correct_leaves()
        # node, reasoning_step = tree.inference()
        # print(reasoning_step)
        # print("*" * 80)
        # print(node.get_trajectory())
        # for child in node.children:
        #     print(child.obtain_reasoning_steps())
        #     print("*" * 80)
        # for each in correct_path:
        #     print(each)
        #     print("*" * 80)
        # client.write(config.data, "Evol_Instruct/config/default_config.json",
                    #  indent=2)
        # print(node.max_depth())
        # break