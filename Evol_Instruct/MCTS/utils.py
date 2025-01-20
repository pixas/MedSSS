import re 

# def extract_template(string, template):
#     pattern = r"The {} is (.*?)(?:\.|$)".format(template)
#     matches = list(re.finditer(pattern, string, re.IGNORECASE))
#     if matches:
#         return matches[-1].group(1).strip()
#     else:
#         return None
def extract_template(string, template):
    # 修改正则表达式模式，使用贪婪匹配 .* 来匹配到最后一个句点
    pattern = r"The {} is(.*)(?:\.|$)".format(template)
    
    # 使用 re.finditer 查找所有匹配项
    matches = list(re.finditer(pattern, string, re.IGNORECASE))
    
    # 如果有匹配项，返回最后一个匹配项的捕获组
    if matches:
        return matches[-1].group(1).strip().strip(".")
    else:
        return None

def parse_action_params(action_name, config):
    if action_name == 'Medrag':
        params = config.Medrag.__dict__
        
        return params
    else:
        return {}




action_stop_words_mapping = {
    "Reason": "</logical_step>",
    "Reflect": "</reflect>",
}

if __name__ == "__main__":
    text = """Extract the text that indicates the dose of the drug.
The text that indicates the dose of insulin is "4 times per day."

However, as insulin is typically prescribed in units and not doses in the classical sense, we can infer that the patient is taking a certain number of units of insulin 4 times per day. Unfortunately, the exact dose (number of units) is not specified in the sentence.

Therefore, the answer is: \"4 times per day\""""
    out = extract_template(text, "answer")
    print(f"**{out}**")