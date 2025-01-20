from Evol_Instruct.actions.base_action import AgentAction, BaseAction
from Evol_Instruct.prompts.prompt_utils import PROMPT_TOKENS, DEFAULT_PROMPT
import json 
import os 


class PromptGen:
    """Prompt Generator Class"""

    def __init__(self) -> None:
        self.prompt_type = "BasePrompt"
        self.examples: dict[str, list] = {}

    # def add_example(
    #     self,
    #     task: TaskPackage,
    #     action_chain: List[tuple[AgentAction, str]],
    #     example_type: str = "action",
    # ):
    #     example_context = task_chain_format(task, action_chain)
    #     if example_type in self.examples:
    #         self.examples[example_type].append(example_context)
    #     else:
    #         self.examples[example_type] = [example_context]

    # def __get_example__(self, example_type: str, index: int = -1):
    #     if example_type in self.examples:
    #         return self.examples[example_type][index]
    #     else:
    #         return None
    
    # def __get_examples__(self, example_type: str) -> str:
    #     """get multiple examples for prompt"""
    #     # check if example_type exist in self.examples
    #     if not example_type in self.examples:
    #         return None
    #     else:
    #         indices = list(range(len(self.examples[example_type])))
    #         examples = [self.__get_example__(example_type, idx) for idx in indices]
    #         return "\n".join(examples)
    

class TaskPromptGen(PromptGen):
    def __init__(
        self,
        action_role: str = 'Think',
        constraint: str = DEFAULT_PROMPT["agent_constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],

    ):
        """Prompt Generator for Base Agent
        :param agent_role: the role of this agent, defaults to None
        :type agent_role: str, optional
        :param constraint: the constraint of this agent, defaults to None
        :type constraint: str, optional
        """
        super().__init__()
        self.prompt_type = "BaseAgentPrompt"
        self.action_role = action_role
        self.constraint = constraint
        self.instruction = instruction
        
    
    def __act_doc_prompt__(self, actions: list[BaseAction], params_doc_flag=True):
        if params_doc_flag:  # given the parameters as the document
            action_doc = [
                {
                    "name": act.action_name,
                    "description": act.action_desc,
                    "parameters": act.params_doc,
                }
                for act in actions
            ]
        else:
            action_doc = {act.action_name: act.action_desc for act in actions}
        
        action_doc = "\n".join([json.dumps(action) for action in action_doc])
        
        prompt = f"""{PROMPT_TOKENS["action"]['begin']}\n{action_doc}\n{PROMPT_TOKENS["action"]['end']}"""
        return prompt