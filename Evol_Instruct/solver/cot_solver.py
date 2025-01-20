from Evol_Instruct.evaluation.eval_em import extract_answer_content
from Evol_Instruct.models.vllm_support import chat_prompt, vllm_clean_generate
from Evol_Instruct.solver.base_solver import Solver
from Evol_Instruct.utils.utils import AlpacaTaskItem
from Evol_Instruct.evaluation.generate_utils import infer_answer

class CoTSolver(Solver):
    def generate_response(self, items: list[AlpacaTaskItem], **kwargs):
        prompts = [chat_prompt([item.prompt], self.server.tokenizer,
                               system='You are a helpful medical assistant.')[0] for item in items]
        temperature = kwargs.pop("temperature", 0)
        n = kwargs.pop("n", 1)
        max_tokens = kwargs.pop("max_tokens", 1024)
        top_p = kwargs.pop("top_p", 0.95)
        
        outputs = self.server(
            prompts,
            wrap_chat=False,
            system='You are a helpful medical assistant.',
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_p=top_p
        )
        outputs = [output[0] for output in outputs]
        generated_answer = [extract_answer_content(x) for x in outputs]
        no_answer_index = [i for i, x in enumerate(generated_answer) if x is None]
        no_answer_outputs = [outputs[i] for i in no_answer_index]
        no_answer_prompts = [prompts[i] for i in no_answer_index]
        if no_answer_index:
            answer_outputs = self.infer_answer(
                no_answer_prompts,
                no_answer_outputs,
                choices_word=self.choices_word,
                cot_prompt=self.cot_prompt
            )
            for i, index in enumerate(no_answer_index):
                outputs[index] = answer_outputs[i]
        answer_outputs = outputs
        # if "The answer is" in 
        # answer_outputs = self.infer_answer(prompts, outputs, self.choices_word,
                                        #    cot_prompt="\nThe answer is ")
        for i, item in enumerate(items):
            item.text = answer_outputs[i]
            
        return items