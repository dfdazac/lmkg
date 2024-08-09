import re
import json

from .tools import tools_dict

def match_tool_call(text: str) -> re.Match:
    """Check if a string contains any tool call.

    Args:
        text: The string to check.
    """
    function_regex = r"<function=(\w+)>(.*?)</function>"
    return re.search(function_regex, text)

def run_tool_call(match: re.Match) -> str:
    """Run a tool call from a regular expression match parsed from a string.

    Args:
        match: The match containing the function and arguments.
    """
    function_name, args_string = match.groups()
    try:
        args = json.loads(args_string) if args_string else {}
        return str(tools_dict[function_name](**args))
    except json.JSONDecodeError as error:
        return f"Error parsing function arguments: {error}"
