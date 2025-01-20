import json

from Evol_Instruct.MCTS.tree_node import MedMCTSNode
from Evol_Instruct.models.vllm_support import VLLMServer
from Evol_Instruct.prompts.prompt_template import mcts_prompts

DEF_INNER_ACT_OBS = "OK"
INNER_ACT_KEY = "response"

class BaseAction:
    def __init__(self, name: str, desc: str):
        self.action_name = name 
        self.description = desc
    
    def __repr__(self):
        string = json.dumps(self.__dict__)
        return string
    
    def call_prompt(self, node: MedMCTSNode):
        raise NotImplementedError

    def __call__(self, *args: json.Any, **kwds: json.Any) -> json.Any:
        raise NotImplementedError

class Reason(BaseAction):
    def __init__(self) -> None:
        action_name = "Reason"
        action_desc = "This action is conducted if the task requires further one-step reasoning or no previous reasoning steps available. "

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
        )
        self.details = "Review all prior reasoning steps and identify the **next single reasoning step** to continue progressing the analysis or deduction. Generate this next step in a clear, concise manner that aligns with the problem's intent and adds relevant insight or depth. Follow the format in <steps> block to provide the next reasoning step."
    
    def call_prompt(self, node: MedMCTSNode) -> str:
        prompt = self.details
        prompt = mcts_prompts['prefix'] + prompt + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" 
        return prompt

    def __call__(self, node: MedMCTSNode, server: VLLMServer, **kwargs):
        prompt = self.call_prompt(node)
        output = server(prompt, **kwargs)
        return output

class Finish(BaseAction):
    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = """This action is conducted if all reasoning steps are complete enough and a final answer can be provided. """

        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
        )
        self.details = """Based on all previous reasoning steps, directly conclude the task by stating: "The answer is {{answer}}". Ensure that this answer is consistent with the prior steps and fully addresses the task question. Do not output other information."""

    def call_prompt(self, node: MedMCTSNode) -> str:
        prompt = self.details
        prompt = mcts_prompts['prefix'] + prompt + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" 
        return prompt

    def __call__(self, node: MedMCTSNode, server: VLLMServer, **kwargs):
        prompt = self.call_prompt(node)
        output = server(prompt, **kwargs)
        return output

class Think(BaseAction):
    def __init__(self, available_actions: list[BaseAction]) -> None:
        action_name = "Think"
        action_desc = "Given all previous reasoning steps, decide on the most suitable action to take next for solving the given problem"
        
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,

        )
        self.available_actions = available_actions
        # self.actions = [eval(f"{action}()") for action in self.available_actions]
        self.action_description = [act.description for act in self.available_actions]
        self.details = f"""Output your choice using the format: "The action is {{action}}." {{action}} can only takes in {self.available_actions}. Do not output other information."""
    
    def call_prompt(self, node: MedMCTSNode) -> str:
        action_docs = [f"{i}. {act.action_name}: {act.description}" for i, act in enumerate(self.available_actions)]
        action_docs = "\n".join(action_docs)
        prompt = self.description + "\n\n" + action_docs + "\n" + self.details
        
        prompt = mcts_prompts['prefix'] + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" + prompt
        return prompt
        

    def __call__(self, node: MedMCTSNode, server: VLLMServer, **kwargs):
        prompt = self.call_prompt(node)
        output = server(prompt, **kwargs)
        return output


class Simulation(BaseAction):
    def __init__(self) -> None:
        action_name = "Simulation"
        action_desc = "This action is conducted if the task requires a simulation of a specific scenario or process. "
        params_doc = {
            INNER_ACT_KEY: """Given all previous reasoning steps, generate the final answer. Conclude the task by stating: "The answer is {{answer}}". Ensure that this answer is consistent with the prior steps and fully addresses the task question."""
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def call_prompt(self, node: MedMCTSNode) -> str:
        prompt = self.params_doc[INNER_ACT_KEY]
        prompt = mcts_prompts['prefix'] + prompt + f"<problem>\n{node.problem}\n</problem>\n\n" + f"<steps>\n{node.obtain_reasoning_steps()[0]}\n</steps>\n\n" 
        return prompt

    def __call__(self, node: MedMCTSNode, server: VLLMServer, **kwargs):
        prompt = self.call_prompt(node)
        output = server(prompt, **kwargs)
        return output
    
