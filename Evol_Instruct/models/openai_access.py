import openai
from openai import OpenAI
import time
import os 
import requests
import concurrent.futures

openai.api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def get_oai_completion(prompt, model, **kwargs):
    temperature = kwargs.get("temperature", 1)
    if model == 'gpt-3.5-turbo':
        max_tokens = kwargs.get("max_tokens", 4096)
    else:
        max_tokens = kwargs.get("max_tokens", 8192)
    top_p = kwargs.get("top_p", 0.95)
    frequency_penalty = kwargs.get("frequency_penalty", 0)
    presence_penalty = kwargs.get("presence_penalty", 0)
    stop = kwargs.get("stop", None)
    n = kwargs.get("n", 1)
    try: 
        completion = client.chat.completions.create(
            model=model,
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                
                ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n = n
        )
        
        res = [x.message.content for x in completion.choices]
       
        gpt_output = res
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.BadRequestError as e:
        # Handle the invalid request error here
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
#             time.sleep(3)
            return get_oai_completion(prompt, model, **kwargs)            
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            return None
    except openai.RateLimitError as e:
        return get_oai_completion(prompt, model)

def call_chatgpt(model, ins, **kwargs):
    success = False
    re_try_count = 15
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = None 
            while ans is None:
                ans = get_oai_completion(ins, model, **kwargs)
                time.sleep(1)
            success = True
        except:
            time.sleep(5)
            print('retry for sample:', ins)
    return ans


def batch_call_chatgpt(model, instructions, max_workers=16, **sampling_params):
    results = [None] * len(instructions)  # 预分配一个与指令长度相同的结果列表
    
    # 定义任务函数，包装call_chatgpt
    def task(instruction):
        return call_chatgpt(model, instruction, **sampling_params)

    # 使用ThreadPoolExecutor来处理多线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，并保留每个future的索引
        futures = {executor.submit(task, instruction): i for i, instruction in enumerate(instructions)}
        
        # 获取所有任务的结果，并根据索引放入results中
        for future in concurrent.futures.as_completed(futures):
            try:
                # 按照原始索引顺序存储结果
                index = futures[future]
                results[index] = future.result()
            except Exception as e:
                print(f"Error processing instruction: {e}")
    
    return results