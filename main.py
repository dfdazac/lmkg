from transformers import pipeline, TextStreamer
import torch

from lmkg.templates import get_template
from lmkg.utils import match_tool_call, run_tool_call
from lmkg.tools import tools_dict


pipe = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cpu",
)
chat_template = get_template("llama3-custom-answer")

messages = []
streamer = TextStreamer(pipe.tokenizer, skip_prompt=False)

while True:
    user_input = input("> ")  # "What time is it?"  # How much is 34 times 546?"
    messages.append({"role": "user", "content": user_input})

    done = False
    while not done:
        inputs = pipe.tokenizer.apply_chat_template(
            messages,
            tools=tools_dict.values(),
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = pipe(
            inputs,
            max_new_tokens=2048,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
            streamer=streamer,
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
