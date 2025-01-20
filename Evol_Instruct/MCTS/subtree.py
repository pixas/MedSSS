from collections import defaultdict, Counter

from regex import F

from Evol_Instruct.MCTS.tree_node import MedMCTSNode
from Evol_Instruct.MCTS.utils import extract_template
from Evol_Instruct.evaluation.generate_utils import set_tokenizer
from Evol_Instruct.MCTS.tree import MCTS

from Evol_Instruct.MCTS.tree_register import tree_registry, register_tree
from Evol_Instruct.actions.base_action import Decompose

@register_tree('SubMCTS')
class SubMCTS(MCTS):
    def __init__(self, item, model_server, config, value_function=None, training=True):
        super().__init__(item, model_server, config, value_function, training)
        self.decompose = Decompose()
    
    def normal_expand_node(self, node, max_children=3, **sampling_params):
        sampling_params['n'] = max_children
        # cur_action = 'Think'
        think_action = self.think_action
        # step_count = len(node.trace)
        if node.depth != 0:
            if node.type == 'Medrag':
                next_actions = ['Reason'] * max_children
            else:
                next_actions = ['Reason'] * max_children
                # next_actions = think_action(node, self.model_server, **sampling_params)
        # pdb.set_trace()
        else:
            next_actions = ['Decompose'] * max_children
        if not next_actions:
            next_actions = ['Reason']
        if node.depth > 0:
            plan = self.root.children[node.trace[1]].reasoning_step 
            
                    # cur_plan = plan.split("\n")[node.depth-2].split(".")[1]
            total_plan_length = len(plan.split("\n"))
            if node.depth == total_plan_length + 1:
                next_actions = ['Finish']
            # next_actions = think_action(node, self.model_server, **sampling_params)
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
                new_node = MedMCTSNode(node.problem, step.strip(), i, parent=node, type=next_actions[i], ground_truth=node.ground_truth,)
                node.add_child(new_node)
    
    def step_observation(self, node, action_list, **sampling_params):
        
        
        
        observations = [None for _ in range(len(action_list))]
        # obtain action_list count
        action_count = Counter(action_list)
        action_index = defaultdict(list)
        for i, action in enumerate(action_list):
            action_index[action].append(i)
        for action, index in action_index.items():
            count = action_count[action]
            if action == 'Decompose':
                pre_action = self.decompose
                sampling_params['n'] = 1
                sub_questions = pre_action(node, self.model_server, first=self.model_server.lora_request is None, question_num=count, few_shot=self.few_shot, **sampling_params)
                output = []
                output = sub_questions
                # action_class = self.actions[action]
                # for j in range(count):
                #     text = action_class(node, self.model_server, few_shot=self.few_shot, first=self.model_server.lora_request is None, pre_gen_texts=sub_questions[j], **sampling_params)
                #     output.append(sub_questions[i] + " " + text.strip())
            else:
            
                sampling_params['n'] = count
                action_class = self.actions[action]
                plan = self.root.children[node.trace[1]].reasoning_step 
                
                cur_plan = plan.split("\n")[node.depth-1].split(".")[1] + ". "
                output = action_class(node, self.model_server, few_shot=self.few_shot, first=self.model_server.lora_request is None, pre_gen_texts=cur_plan, **sampling_params)
                output = [cur_plan + " " + text.strip() for text in output]
            # pdb.set_trace()
            
            for i in range(len(output)):
                observations[index[i]] = output[i].strip()
            # else:
            #     print(f"{action} is not supported in the current action space: {self.base_actions}")
            #     raise NotImplementedError

            # pdb.set_trace()
        return observations
    
    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from peft import PeftModel
    import torch 
    from Evol_Instruct import logger, client
    from Evol_Instruct.models.vllm_support import VLLMServer
    from Evol_Instruct.MCTS.tree import MCTSConfig
    import time
    torch.set_default_device("cuda")
    model_path = '/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct'
    # model_path = '/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3.1-8B-Instruct'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = set_tokenizer(tokenizer)
    # data = client.read("s3://syj_test/datasets/medical_train/pubhealth.json")
    # data = client.read("s3://syj_test/datasets/medical_test/MedQA_cot.json")
    data = client.read("s3://syj_test/datasets/medical_train/mmed_en_train.json")
    # for i in range(15):
    # model_base = LlamaForValueFunction.from_pretrained(
    #     "/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct",
    #     num_labels=1
    # )
    # value_model = PeftModel.from_pretrained(model_base, "/mnt/petrelfs/jiangshuyang.p/checkpoints/llama38b_mcts_vllm_mmed_en_train_all_trial5/sft_combined_1_new-llama3-8b-r16a32-1epoch-VALUE-ITER1")
    # value_model = value_model.merge_and_unload().to(torch.float16)
    # value_model = model_base
    value_model = None
    server = VLLMServer(url="http://10.140.1.163:10002", model=model_path, tokenizer=tokenizer, offline=False)
    # server = VLLMServer(url="http://10.140.1.163:10002", model=model_path, tokenizer=tokenizer, offline=True, lora_path='/mnt/petrelfs/jiangshuyang.p/checkpoints/llama38b_mcts_vllm_mmed_en_train_all_trial5/sft_combined_1_new-llama3-8b-r16a32-1epoch-SFT-ITER1', gpu_memory_usage=0.45, max_model_len=8192)
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
        config = client.read("Evol_Instruct/config/submcts.json")
        config = MCTSConfig(config)
        # config.finish_uncompleted = True
        # config.max_children=3
        
        # config.max_children = 6
        mcts_cls = tree_registry[config.mcts_cls]
        
        tree = mcts_cls(item, model_server=server, config=config, value_function=value_model, training=True)
        tokenizer = tree.tokenizer 
        start = time.time()
        # node = tree.tot_inference()
        root = tree.run(temperature=0.7)
        # node, node_steps = tree.inference("vote-sc")
        # exit(0)
        correct_leaves = tree.obtain_correct_leaves()
        incorrect_leaves = tree.obtain_incorrect_leaves()
        repeat_try = config.repeat_try
        while not correct_leaves and repeat_try > 0:
            tree = mcts_cls(item, model_server=server, config=config, value_function=value_model, training=True)
            root = tree.run()
            correct_leaves = tree.obtain_correct_leaves()
            repeat_try -= 1
        end = time.time()
        # print(node.obtain_reasoning_steps()[0], flush=True)
        # pdb.set_trace()
        client.write("", "debug.log", mode='w')
        fp = open("debug.log", 'a')
        leaves = tree.obtain_leaves(root)
        for leaf in leaves:
            print(leaf.obtain_reasoning_steps()[0], leaf.value_chain(), file=fp, flush=True)
            print("*" * 100, file=fp, flush=True)
        
        logger.info(f"Time cost: {end - start}")
        