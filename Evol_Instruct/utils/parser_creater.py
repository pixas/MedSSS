import argparse
from vllm import SamplingParams
import inspect
from typing import get_type_hints, Optional, Union, List, Any, Set
import pdb

def get_general_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # data I/O arguments
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output_data', type=str, required=True, help='Path to the output data file')
    parser.add_argument('--sample_num', type=int, default=-1, help='Number of samples to process')
    parser.add_argument("--sample_idx", type=int, default=-1)
    parser.add_argument('--resume', action='store_true', help='Resume from the last processed item')
    parser.add_argument('--resume_merge_file', action="store_true", help="Resume from the last merged file")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument('--clean_outputs', action='store_true', help='Clean the output file before writing')
    parser.add_argument('--debug', action='store_true', help='Whether to output debugging info')
    # Data synthesization arguments
    parser.add_argument('--model', type=str, required=True, help='Model to use for generating instructions')
    parser.add_argument("--prompt_type", type=str, default='general')
    parser.add_argument("--num_process", default=1, type=int, help="Number of processes to use")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--max_rounds', type=int, default=3, help='Maximum number of rounds for multi-round conversation')
    parser.add_argument('--iteration', type=int, default=1, help='Number of iterations for each sample')
    parser.add_argument('--evol_path', type=str, choices=['all', 'random'], help='evolve path; all for five directions and random means randomly choose one from five directions')
    return parser

def add_argument_to_parser(parser, name, param_type, help_text=None):
    # Map Python types to argparse types
    type_map = {
        int: int,
        float: float,
        str: str,
        bool: bool,
        list: list,
        Optional[int]: int,
        Optional[float]: float,
        Optional[str]: str,
        Optional[list]: list,
        Union[bool, str]: str,  # Treat as str to handle both types
    }

    # Determine the action or type for booleans
    if param_type == bool:
        parser.add_argument(f'--{name}', action='store_true', required=False, help=help_text)
    else:
        parser.add_argument(f'--{name}', type=type_map.get(param_type, str), required=False, help=help_text)

def create_argument_parser(parent_parser=None) -> argparse.ArgumentParser:
    if parent_parser is not None:
        parser = argparse.ArgumentParser(description="Parse SamplingParams arguments.", parents=[parent_parser,])
    else:
        parser = argparse.ArgumentParser(description="Parse SamplingParams arguments.")
    
    # Get the type hints for class attributes
    param_hints = get_type_hints(SamplingParams)

    # Get the docstring of the class
    class_doc = inspect.getdoc(SamplingParams)
    
    # Extract the argument descriptions from the class docstring
    arg_help = {}
    if class_doc:
        lines = class_doc.split("\n")
        current_arg = None
        i = 0
        total_lines = len(lines)
        begin_process = False
        while i < total_lines:
            line = lines[i].strip()
            if line.startswith("Args:"):
                i += 1
                begin_process = True
                continue
            if line and ":" in line and begin_process:
                arg_name, help_line = line.split(":")
                current_arg = arg_name.strip()
                arg_help[current_arg] = help_line.strip()
                # pdb.set_trace()
                i += 1
                while i < total_lines and (":" not in lines[i].strip() or (":" in lines[i] and " " in lines[i].split(":")[0].strip())):
                    # i += 1
                    arg_help[current_arg] += lines[i].strip()
                    i += 1
            else:
                i += 1
        # for line in lines:
        #     line = line.strip()
        #     if line.startswith("Args:"):
        #         continue
        #     if line and ":" in line:
        #         arg_name, help_line = line.split(":")
        #         current_arg = arg_name.strip()
        #         arg_help[current_arg] = help_line.strip()
        #     elif current_arg:
        #         arg_help[current_arg] += line

    # Loop over class attributes and dynamically add them to the argument parser
    for name, param_type in param_hints.items():
        help_text = arg_help.get(name, f"Argument for {name}.")
        add_argument_to_parser(parser, name, param_type, help_text)

    return parser

def get_sampling_parser():
    sampling_parser = create_argument_parser()
    # generate_args = 
    # generate_parser.print_help()
    # args = sampling_parser.parse_args()
    # # for action in generate_parser._actions:
    # #     print(f"Argument: {action.option_strings}, Help: {action.help}")
    # return args
    return sampling_parser

if __name__ == "__main__":
    args = get_sampling_parser()
    kwargs = {key: value for key, value in args._get_kwargs() if value is not None and (value)}
    print(kwargs)