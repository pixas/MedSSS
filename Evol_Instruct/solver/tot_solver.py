from copy import deepcopy
from Evol_Instruct.MCTS.utils import extract_template
from Evol_Instruct.evaluation.eval_em import extract_answer_content
from Evol_Instruct.models.vllm_support import VLLMServer, chat_prompt, vllm_clean_generate
from Evol_Instruct.solver.base_solver import Solver
from Evol_Instruct.utils.utils import AlpacaTaskItem, extract_answer, compute_weighted_values
from collections import Counter, defaultdict 
from itertools import chain
import numpy as np
import pdb
import torch
from Evol_Instruct.solver.sc_vm_solver import obtain_prm_value_for_single_pair


class ToTSolver(Solver):
    def __init__(self, *args, value_model, infer_rule, **kwargs):
        """
        server: VLLMServer, a server to handle model inference
        save_file: TextIOWrapper, a file handler
        choices_word: list, a list of choice words
        cot_prompt: str, the prompt to infer answer
        
        """
        super().__init__(*args, **kwargs)
        self.value_model = value_model
        self.infer_rule = infer_rule 
    
    
    def generate_response(self, items: list[AlpacaTaskItem], **kwargs):
        temperature = kwargs.pop("temperature", 1.0)
        n = kwargs.pop("n", 16)
        max_tokens = kwargs.pop("max_tokens", 1024)
        top_p = kwargs.pop("top_p", 0.95)
        
        
        