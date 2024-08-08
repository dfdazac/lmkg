from transformers import pipeline, TextStreamer
import torch
from datetime import datetime
import json
import math

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

pipe = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
PYTHON_TOKEN = pipe.tokenizer.get_added_vocab()['<|python_tag|>']
streamer = TextStreamer(pipe.tokenizer, skip_prompt=False)

messages = [{"role": "system",
             "content": "You are a helpful assistant with tool calling capabilities. "
                        "When you receive a tool call response, use the output to format an answer to the original user question. "
                        "If multiple tool calls are needed, only make one at a time and wait for the tool response."}]
while True:
    user_input = "How much is 34 times 546?"
    messages.append({"role": "user", "content": user_input})

    done = False
    while not done:
        inputs = pipe.tokenizer.apply_chat_template(
            messages,
            tools=tools.values(),
            tokenize=False,
            add_generation_prompt=True
        )

        output_ids = pipe(
            inputs,
            max_new_tokens=2048,
            pad_token_id=pipe.tokenizer.eos_token_id,
            streamer=streamer,
            return_tensors=True,
        )[0]['generated_token_ids']

        input_ids = pipe.tokenizer(inputs)['input_ids']
        input_length = len(input_ids)
        output_ids = output_ids[input_length:]

        outputs = pipe.tokenizer.decode(output_ids)
        messages.append({"role": "assistant", "content": outputs})

        if output_ids[0] == PYTHON_TOKEN:
            function_call = pipe.tokenizer.decode(output_ids, skip_special_tokens=True)
            function_dict = json.loads(function_call)
            # print(f'\tFunction call: {function_dict}')
            function_output = tools[function_dict["name"]](**function_dict["parameters"])
            function_output = str(function_output)
            # print(f'\tOutput: {function_output}')

            messages.append({"role": "ipython", "content": {"output": function_output}})

        else:
            done = True
            # print(outputs)
            print("=" * 50)

    # break
