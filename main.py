from transformers import pipeline, TextStreamer
import torch
from datetime import datetime
import json
import re

from lmkg.templates import get_template

def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())


def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

tools = {"current_time": current_time,
         "multiply": multiply}

def match_tool_call(text: str) -> re.Match:
    """Check if a string contains any tool call."""
    function_regex = r"<function=(\w+)>(.*?)</function>"
    return re.search(function_regex, text)

def run_tool_call(match: re.Match) -> str:
    function_name, args_string = match.groups()
    try:
        args = json.loads(args_string) if args_string else {}
        return str(tools[function_name](**args))
    except json.JSONDecodeError as error:
        return f"Error parsing function arguments: {error}"

pipe = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
PYTHON_TOKEN = pipe.tokenizer.get_added_vocab()['<|python_tag|>']
streamer = TextStreamer(pipe.tokenizer, skip_prompt=False)
chat_template = get_template("llama3-custom-answer")

messages = []

while True:
    user_input = input("> ")  # "What time is it?"  # How much is 34 times 546?"
    messages.append({"role": "user", "content": user_input})

    done = False
    while not done:
        inputs = pipe.tokenizer.apply_chat_template(
            messages,
            tools=tools.values(),
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = pipe(
            inputs,
            max_new_tokens=2048,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
            streamer=None,
        )[0]['generated_text']

        messages.append({"role": "assistant", "content": outputs})

        if match := match_tool_call(outputs):
            print("=" * 50)
            print("Calling function...")
            print(outputs)
            tool_result = run_tool_call(match)
            print(tool_result)
            messages.append({"role": "ipython", "content": {"output": tool_result}})
            print("=" * 50)
        else:
            print(outputs)
            done = True
