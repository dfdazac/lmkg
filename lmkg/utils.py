import json
import logging
import os
import re
from typing import Iterable

import jinja2

from .tools import Tool


def get_chat_template(name: str):
    """Get content of Jinja template from file"""
    templates_dir = os.path.dirname(__file__)
    template_path = os.path.join(templates_dir, 'templates', f'{name}.jinja')
    with open(template_path, 'r') as f:
        return f.read()


def build_task_input(task: str, task_kwargs: dict):
    env = jinja2.Environment(loader=jinja2.PackageLoader("lmkg",
                                                         "prompts"))
    prompt = env.get_template(f"{task}.jinja")
    return prompt.render(**task_kwargs)


def parse_tool_call(text: str, format: str = "json"):
    """Check if a string contains any tool call and parse its contents.

    Args:
        text: The string to check
        format: The format used to parse the tool call. One of "xml", "json"
    """
    # Check first if a call in either XLM or JSON format is found
    if format == "xml":
        regex = r"<function=(\w+)>(.*?)</function>"
    elif format == "json":
        regex = r'"name":\s*"(\w+)",\s*"parameters":\s*(\{.*?\})'
    else:
        raise ValueError("Format must be either xml or json")

    text = text.replace("\n", "")
    match = re.findall(regex, text)
    match_info = None
    if not match:
        return None

    if len(match) > 1:
        match_info = ("Only one function call is allowed. "
                      "Executing the first one.")

    function_name, args_string = match[0]
    args = json.loads(args_string) if args_string else {}

    return function_name, args, match_info


def run_if_callable(text: str, tools: Iterable[Tool], format: str = "json"):
    """
    Run a tool if the input text contains a call to it. Only the first function
    is executed.

    Args:
         text: The string optionally containing a tool call
         tools: A list of tools to run if a match to their functions is found.
            The order in the list determines priority when matching a function.
         format: The format used to parse the tool call. One of "xml", "json"
    """
    result, match_info = None, None
    if match := parse_tool_call(text, format):
        function_name, function_args, match_info = match

        function = None
        result = None
        for tool in tools:
            if hasattr(tool, function_name):
                function = getattr(tool, function_name)
                try:
                    result = function(**function_args)
                except Exception as e:
                    result = str(e)
                break

        if not function:
            result = f"Tool {function_name} not found."

    return result, match_info


def get_logger():
    """Get a default logger that includes a timestamp."""
    logger = logging.getLogger('lmkg')
    logger.handlers = []
    ch = logging.StreamHandler()
    str_fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(str_fmt, datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    return logger
