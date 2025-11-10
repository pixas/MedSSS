
from Evol_Instruct.models.vllm_support import chat_prompt
from Evol_Instruct.solver.base_solver import Solver
from Evol_Instruct.utils.utils import AlpacaTaskItem
from Evol_Instruct.models.openai_access import batch_call_chatgpt

class CoTSolver(Solver):
    def generate_response(self, items: list[AlpacaTaskItem], **kwargs):
        prompts = [chat_prompt([item.prompt], self.server.tokenizer,
                               system=self.system)[0] for item in items]
        temperature = kwargs.pop("temperature", 0)
        n = kwargs.pop("n", 1)
        max_tokens = kwargs.pop("max_tokens", 1024)
        top_p = kwargs.pop("top_p", 0.95)
        
        outputs = self.server(
            prompts,
            wrap_chat=False,
            # system='You are a helpful medical assistant.',
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_p=top_p
        )
        outputs = [output[0] for output in outputs]
        if self.whether_infer_answer:
            answer_outputs = self.infer_answer(
                prompts,
                outputs,
                choices_word=self.choices_word,
                cot_prompt=self.cot_prompt
            )
        else:
            answer_outputs = outputs

        for i, item in enumerate(items):
            item.text = answer_outputs[i]
            
        return items



class GPTCoTSolver(Solver):
    def generate_response(self, items: list[AlpacaTaskItem], **kwargs):
        prefix_prompt = "Think the problem step by step and finalize the answer as 'The answer is ANS', where `ANS` is the final answer. For multiple-choice problem, ANS should be the choice letter.\n\n"
        prompts = [prefix_prompt + item.prompt for item in items]
        # prompts = [chat_prompt([item.prompt], self.server.tokenizer,
                            #    system=self.system)[0] for item in items]
        temperature = kwargs.pop("temperature", 0)
        n = kwargs.pop("n", 1)
        max_tokens = kwargs.pop("max_tokens", 1024)
        top_p = kwargs.pop("top_p", 0.95)
        # outputs = [
        #     call_chatgpt(self.server, prompt, n=n, max_tokens=max_tokens) for prompt in prompts
        # ]        
        outputs = batch_call_chatgpt(self.server, prompts, n=n, max_tokens=max_tokens)
        
        outputs = [output[0] for output in outputs]
        
        # if "The answer is" in 
        # answer_outputs = self.infer_answer(prompts, outputs, self.choices_word,
                                        #    cot_prompt="\nThe answer is ")
        for i, item in enumerate(items):
            item.text = outputs[i]
            
        return items