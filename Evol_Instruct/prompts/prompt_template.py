

mcts_prompts = {
#     "Planning": """You are tasked with navigating a reasoning task with the following actions available to you at each step:

# Reason: Think through how to proceed to the next step in reasoning based on all prior information.
# Reflect: Review the previous thought process to identify if there were any errors or gaps.
# Finish: When confident with the entire reasoning chain, conclude with the final answer using the format: "The answer is {{answer}}".

# Follow these guidelines to choose the appropriate action at each step, considering the progression of reasoning thus far.""",
    # "Reason": """Given all previous reasoning steps and the problem, take the next **ONE** logical step to continue the analysis or deduction process. Focus on producing only a single step in reasoning, based on all prior information. Ensure this step aligns with the problems' intents and adds clarity or depth to the argument.""",
    "Reason": """Review all prior reasoning steps and identify the **next single reasoning step** to continue progressing the analysis or deduction. Generate this next step in a clear, concise manner that aligns with the problem's intent and adds relevant insight or depth. Follow the format in <steps> block to provide the next reasoning step.""",
    
    "Reflect": """Review the previous reasoning step critically to identify any necessary corrections, refinements, or enhancements. Suggest **one single revision step** to improve the coherence, accuracy, or depth of the previous reasoning. Ensure this revision aligns with the problem's intent and brings clarity or correctness to the analysis. Follow the format in <steps> block to provide the next revised reasoning step.""",
    # "Finish": """Based on all previous reasoning steps, synthesize the entire thought process to arrive at a final answer. Conclude the task by stating: "The answer is {{answer}}". Ensure that this answer is consistent with the prior steps and fully addresses the task question.""",
    "Finish": """Based on all previous reasoning steps, directly conclude the task by stating: "The answer is {{answer}}". Ensure that this answer is consistent with the prior steps and fully addresses the task question. Do not output other information.""",
    "Think": """Given all previous reasoning steps, decide on the most suitable action to take next for solving the given problem:

1. Reason: if the task requires further one-step reasoning or no previous reasoning steps available. Must choose this if no previous reasoning steps are available.
2. Reflect: if the previous reasoning steps contain ambiguity or mistakes. Only choose "Reflect" if there are prior reasoning steps.
3. Finish: if all reasoning steps are complete enough and a final answer can be provided. Only choose "Finish" if there are two or more reasoning steps.
4. Search: if you need to search for more information to continue the reasoning process. Only choose "Search" if the search content is not provided in the problem.
Output your choice using the format: "The action is {{action}}." {{action}} can only takes 'Reason', 'Reflect', 'Search' or 'Finish'. Do not output other information.""",
    'Simulate': """Given all previous reasoning steps, generate the final answer. Conclude the task by stating: "The answer is {{answer}}". Ensure that this answer is consistent with the prior steps and fully addresses the task question.""",
    "prefix": "You are a professional medical expert majored at reasoning in hard medical-related problems.\n\n",
    "hint": "The answer to the problem is {answer}.",
    "ls_prefix": """You are a professional medical assistant. Please think deeply about the following medical problem to derive at the final answer. Conclude the answer using the format: "The answer is {{answer}}.".\n\nProblem: {problem}""",
    "ls_refine_prefix": """You are a professional medical assistant. Please think deeply about the following medical problem to derive at the final answer. Follow the given example to critically rethink the wrong answer and correct it. Begin your correction process with this sentence: '{refine_prompt}'. Conclude the answer using the format: "The answer is {{answer}}.".\n\nProblem: {problem}\n\nWrong: {wrong}"""
}

mcts_prompts['think_template'] = "{planning}\n\nProblem: {problem}\n\nReasoning Steps:\n{steps}\n\n{think}"
# mcts_prompts['action_template'] = mcts_prompts['prefix'] + "Problem: {problem}\n\nReasoning Steps: {steps}\n\n{action}"
# mcts_prompts['action_template'] = mcts_prompts['prefix'] + "{action}\n\n<problem>\n{problem}\n</problem>\n\n<reasoning_steps>\n{steps}\n</reasoning_steps>\n\n"
mcts_prompts['action_template'] = mcts_prompts['prefix'] + "{action}\n\n<problem>\n{problem}\n</problem>\n\n<steps>\n{steps}\n</steps>"
# mcts_prompts['reason_template'] = mcts_prompts['prefix'] + f"{mcts_prompts['Reason']}\n" + "<problem>\n{problem}\n</problem>\n\n<steps>\n{steps}\n\n"

# mcts_prompts['node_template'] = mcts_prompts['prefix'] + "Problem: {problem}\n\nReasoning Steps: {steps}\n\n{action_prompt}"
mcts_prompts['node_template'] = mcts_prompts['prefix'] + "<problem>\n{problem}\n</problem>\n\n<reasoning_steps>\n{steps}\n</reasoning_steps>\n\n{action_prompt}"
mcts_prompts['simu_template'] = mcts_prompts['prefix'] + "Problem: {problem}\n\nReasoning Steps:\n{steps}\n\n{Simulate}"
mcts_prompts['break_down'] = "Let's break down this problem step by step."
mcts_prompts['ls_break_down'] = "Alright, so I'm faced with this medical question. Let's break it down step by step."
mcts_prompts['ls_simu_template'] = mcts_prompts['ls_prefix']


search_prompts = {
    "query_extracter": """Write a concise, targeted query that will help retrieve relevant documents based on the following context and previous steps. The query should focus on addressing the main points in the problem statement and consider any previous actions taken to avoid redundancy.

Inputs:
<problem>\n{problem}\n</problem>
<steps>\n{steps}\n</steps>
Requirements:
1. The query should be clear, specific, and aimed at retrieving information relevant to the <problem>.
2. Avoid aspects that have already been addressed in <steps>.
3. Emphasize terms and concepts that will yield precise results without being overly restrictive.
Output Format:

Query: [Generated Query Here]
Key Focus Points: [Key Concepts or Terms used in the Query]""",
    
}


orm_teacher_prompt = """Please evaluate whether the provided model answer is correct based on the following criteria:

1. First, check if the model's answer matches the reference answer. If it does not match, mark it as incorrect.
2. If the model's answer matches the reference answer, further assess the clarity and accuracy of the reasoning process. If the reasoning is clear, accurate, and logically consistent, mark it as correct; if the reasoning is unclear, inaccurate, or includes steps that do not logically lead to the correct answer, mark it as incorrect.

Problem: {problem}
Model Output: {output}
Reference Answer: {answer}

Please output the evaluation result in JSON format as {"result": true} or {"result": false}.
"""