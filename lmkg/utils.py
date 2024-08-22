import re
import json

def match_tool_call(text: str, format: str = "xml"):
    """Check if a string contains any tool call.

    Args:
        text: The string to check.
        format: The format used to parse the tool call.
    """
    match_info = None
    if format == "xml":
        function_regex = r"<function=(\w+)>(.*?)</function>"
        match = re.findall(function_regex, text)
        if not match:
            return None
        else:
            if len(match) > 1:
                match_info = "Only one function call is allowed. Executing the first one."

            function_name, args_string = match[0]
            args = json.loads(args_string) if args_string else {}
    elif format == "json":
        data = json.loads(text)
        if data.get("type") == "function":
            function_name = data.get("name")
            args = data.get("parameters", {})
    else:
        raise ValueError("Format must be either xml or json")

    return function_name, args, match_info
