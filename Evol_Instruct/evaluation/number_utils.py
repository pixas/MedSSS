# copy from https://github.com/TIGER-AI-Lab/MAmmoTH/blob/main/math_eval/number_utils.py
import re
import math
from latex2sympy2 import latex2sympy
import multiprocessing
import time

from math import sqrt, sin, cos, log, pi, factorial, exp, e
E = 2.718


def run_eval(expression, output):
    try:
        # Safely evaluate the expression
        result = eval(expression)
        output.put(result)
    except Exception as e:
        output.put(e)


def eval_with_timeout(expression, timeout=5):
    # Create a multiprocessing.Queue to receive the output
    output = multiprocessing.Queue()
    
    # Define and start the process
    process = multiprocessing.Process(target=run_eval, args=(expression, output))
    process.start()
    
    # Wait for the process to complete or timeout
    process.join(timeout)
    
    if process.is_alive():
        # Terminate the process
        process.terminate()
        process.join()
        return "Timeout or error during evaluation"
    
    # Get result from the queue
    try:
        return output.get_nowait()
    except Exception as e:
        return "Error retrieving result"


def floatify(num: str):
    try:
        num = float(num)
        if num.is_integer():
            return round(num)
        else:
            return num
    except Exception:
        return None


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def clean_units(pred_str: str):
    """Clean the units in the number."""
    def convert_pi_to_number(code_string):
        code_string = code_string.replace('\\pi', 'π')
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r'(?<![\d}])\\?π', '3.14', code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r'(\d)(\\?π)', r'\1*3.14', code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r'\{(\\?π)\}', '3.14', code_string)
        code_string = re.sub(r'\*(\\?π)', '*3.14', code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace('%', '/100')
    pred_str = pred_str.replace('$', '')
    pred_str = pred_str.replace('¥', '')
    pred_str = pred_str.replace('°C', '')
    pred_str = pred_str.replace(' C', '')
    pred_str = pred_str.replace('°', '')
    return pred_str


def number_it(num):
    if isinstance(num, (int, float)):
        return num

    num = clean_units(num)
    try:
        num = str(latex2sympy(num))
    except Exception:
        pass

    if floatify(num) is not None:
        return floatify(num)
    else:
        try:
            num = eval_with_timeout(num)
            if isinstance(num, list) or isinstance(num, tuple):
                num = num[0]
            if floatify(num) is not None:
                return floatify(num)
            else:
                return None
        except Exception:
            return None


def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        if isinstance(gt, int):
            return round(p) == gt
        else:
            return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    elif len(pred) != len(gt):
        return False
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([compare_two_numbers(p, g) for p, g in zip(pred, gt)])