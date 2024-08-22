import torch
from transformers import TextStreamer, pipeline
from pprint import pprint

from lmkg.tools import GraphDBConnector
from lmkg.utils import match_tool_call, get_template


pipe = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
chat_template = get_template("llama3")

graphdb = GraphDBConnector("http://localhost:7200/repositories/wikidata5m")

messages = []
streamer = TextStreamer(pipe.tokenizer, skip_prompt=False)
printed_system = False

user_input = "Search for the definition of entity Q3308285."
messages.append({"role": "user", "content": user_input})

done = False
while not done:
    inputs = pipe.tokenizer.apply_chat_template(
        messages,
        tools=graphdb.tools_json,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True
    )

    if not printed_system:
        print(inputs)
        printed_system = True

    outputs = pipe(
        inputs,
        max_new_tokens=2048,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
        streamer=None,
    )[0]['generated_text']

    messages.append({"role": "assistant", "content": outputs})

    if match := match_tool_call(outputs, format="json"):
        print("=" * 50)
        print("Calling function...")
        print(outputs)
        function_name, args, match_info = match
        tool_result = getattr(graphdb, function_name)(**args)
        if match_info:
            pprint(match_info)
            messages.append({"role": "ipython", "content": {"output": match_info}})
        pprint(tool_result)
        messages.append({"role": "ipython", "content": {"output": tool_result}})
        print("=" * 50)
    else:
        print(outputs)
        done = True
