import re
import json

def match_tool_call(text: str):
    """Check if a string contains any tool call.

    Args:
        text: The string to check.
    """
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, text)
    if not match:
        return None
    else:
        function_name, args_string = match.groups()
        args = json.loads(args_string) if args_string else {}
        return function_name, args
