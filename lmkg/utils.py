import re
import json

def match_tool_call(text: str, format: str = "xml"):
    """Check if a string contains any tool call.

    Args:
        text: The string to check.
        format: The format used to parse the tool call. One of "xml", "json".
    """
    if format == "xml":
        regex = r"<function=(\w+)>(.*?)</function>"
    elif format == "json":
        regex = r'"name":\s*"(\w+)",\s*"parameters":\s*(\{.*?\})'
    else:
        raise ValueError("Format must be either xml or json")

    match = re.findall(regex, text)
    match_info = None
    if not match:
        return None
    else:
        if len(match) > 1:
            match_info = ("Only one function call is allowed. "
                          "Executing the first one.")

        function_name, args_string = match[0]
        args = json.loads(args_string) if args_string else {}

    return function_name, args, match_info
