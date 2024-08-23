import torch
from transformers import pipeline, BitsAndBytesConfig
from pprint import pprint
import jinja2
from typing import Literal

from tap import Tap

from lmkg.tools import GraphDBTool, AnswerStoreTool
from lmkg.utils import match_tool_call, get_template
from utils import LlamaModels, MODELS, get_model


class Arguments(Tap):
    model: MODELS = LlamaModels.LLAMA_31_8B
    quantization: Literal["8bit", "4bit"] = None
    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"

def main(args: Arguments):
    pipe = get_model(args.model, args.quantization)
    chat_template = get_template("llama3-kg")

    graphdb_tool = GraphDBTool(args.graphdb_endpoint)
    answer_tool = AnswerStoreTool()

    messages = []
    printed_system = False

    env = jinja2.Environment(loader=jinja2.PackageLoader("lmkg",
                                                         "prompts"))
    prompt = env.get_template("entity_linking.jinja")

    text = ("Amsterdam is the Netherlands’ capital, known for its artistic "
            "heritage elaborate canal system and narrow houses with gabled "
            "facades, legacies of the city’s 17th-century Golden Age. Its"
            "Museum District houses the Van Gogh Museum, works by Rembrandt "
            "and Vermeer at the Rijksmuseum, and modern art at the Stedelijk.")
    messages.append({"role": "user", "content": prompt.render(text=text)})

    done = False
    while not done:
        inputs = pipe.tokenizer.apply_chat_template(
            messages,
            tools=graphdb_tool.tools_json + answer_tool.tools_json,
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

            tool = None
            if hasattr(graphdb_tool, function_name):
                tool = getattr(graphdb_tool, function_name)
            elif hasattr(answer_tool, function_name):
                tool = getattr(answer_tool, function_name)

            if tool:
                tool_result = tool(**args)
            else:
                tool_result = f"Tool {function_name} not found."
            if match_info:
                pprint(match_info)
                messages.append({"role": "ipython", "content": {"output": match_info}})
            pprint(tool_result)
            messages.append({"role": "ipython", "content": {"output": tool_result}})
            print("=" * 50)
        else:
            print(outputs)
            done = True

    print("*" * 50)
    print(text)
    print(answer_tool.answer)


main(Arguments().parse_args())
