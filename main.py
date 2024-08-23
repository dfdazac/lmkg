from pprint import pprint
from typing import Literal

import jinja2
from tap import Tap

from lmkg.tools import AnswerStoreTool, GraphDBTool
from lmkg.utils import get_template, match_tool_call
from utils import MODELS, LlamaModels, get_model


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
    prompt = env.get_template("contradiction_generation.jinja")

    passage = "Amsterdam is the capital of the Netherlands."
    triples = "[[Q727, P1376, Q55]]"
    messages.append({"role": "user",
                     "content": prompt.render(passage=passage,
                                              triples=triples)})

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
    print(answer_tool.answer)


main(Arguments().parse_args())
