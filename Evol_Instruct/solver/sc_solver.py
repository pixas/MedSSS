from Evol_Instruct.evaluation.eval_bench import extract_answer_content
from Evol_Instruct.models.vllm_support import chat_prompt
from Evol_Instruct.solver.base_solver import Solver
from Evol_Instruct.utils.utils import AlpacaTaskItem
from collections import Counter 



class SCSolver(Solver):
    def generate_response(self, items: list[AlpacaTaskItem], **kwargs):
        temperature = kwargs.pop("temperature", 1.0)
        n = kwargs.pop("n", 16)
        max_tokens = kwargs.pop("max_tokens", 1024)
        top_p = kwargs.pop("top_p", 0.95)
        
        prompts = [chat_prompt([item.prompt], self.server.tokenizer,
                               system='You are a helpful medical assistant.')[0] for item in items]
        
        outputs = self.server(
            prompts,
            wrap_chat=False,
            system='You are a helpful medical assistant.',
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_p=top_p   
        )
        outputs = [y for x in outputs for y in x]
        
        generated_answer = [extract_answer_content(x) for x in outputs]
        no_answer_index = [i for i, x in enumerate(generated_answer) if x is None]
        
        no_answer_outputs = [outputs[i] for i in no_answer_index]
        no_answer_prompts = [prompts[i // n] for i in no_answer_index]
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

        batch = len(items)
        outputs = [[answer_outputs[i * n + j] for j in range(n)] for i in range(batch)] 
        
        batch_answer_content = [[extract_answer_content(x)[0].replace('"', '').replace("'", "").lower() for x in y] for y in outputs]
        batch_answer_index = [row.index(Counter(row).most_common(1)[0][0]) for row in batch_answer_content]

        # print(batch_answer_content)
        outputs = [y[x] for x, y in zip(batch_answer_index, outputs)]
        for i, item in enumerate(items):
            item.text = outputs[i]
            item.all_answer = batch_answer_content[i]
        
        
        return items
