from pprint import pprint
from typing import Literal

import jinja2
import torch
from huggingface_hub import InferenceClient
from tap import Tap

from lmkg.tools import AnswerStoreTool, GraphDBTool
from lmkg.utils import get_template, match_tool_call
from utils import MODELS, LlamaModels, get_model_and_tokenizer


class Arguments(Tap):
    model: MODELS = LlamaModels.LLAMA_31_8B
    inference_client: str = None
    quantization: Literal["8bit", "4bit"] = None
    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"


def main(args: Arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, gen_config = get_model_and_tokenizer(
        args.model,
        args.quantization,
        skip_model=args.inference_client is not None
    )
    if model is not None:
        model = model.to(device)
    if args.inference_client:
        client = InferenceClient(args.inference_client)

    chat_template = get_template("llama3-kg")

    graphdb_tool = GraphDBTool(args.graphdb_endpoint)
    answer_tool = AnswerStoreTool()

    messages = []
    printed_system = False

    env = jinja2.Environment(loader=jinja2.PackageLoader("lmkg",
                                                         "prompts"))
    prompt = env.get_template("entity_linking.jinja")

    text = "Amsterdam is the capital of the Netherlands."
    messages.append({"role": "user",
                     "content": prompt.render(text=text)})

    done = False
    while not done:
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=graphdb_tool.tools_json + answer_tool.tools_json,
            chat_template=chat_template,
            tokenize=args.inference_client is None,
            add_generation_prompt=True,
            return_dict=args.inference_client is None,
            return_tensors="pt"
        )

        if not args.inference_client:
            inputs = inputs.to(device)
            input_length = inputs['input_ids'].shape[-1]
            outputs = model.generate(**inputs,
                           max_new_tokens=2048,
                           pad_token_id=tokenizer.eos_token_id)[0]

            outputs = tokenizer.decode(outputs[input_length:])
        else:
            outputs = client.text_generation(
                inputs,
                return_full_text=False,
                max_new_tokens=2048,
                do_sample=gen_config.do_sample,
                temperature=gen_config.do_sample,
                top_k=gen_config.top_k,
                top_p=gen_config.top_p,
                details=False
            )

        if not printed_system:
            if not args.inference_client:
                inputs = tokenizer.decode(inputs['input_ids'][0])
            print(inputs)
            printed_system = True

        messages.append({"role": "assistant", "content": outputs})

        if match := match_tool_call(outputs, format="json"):
            print("=" * 50)
            print("Calling function...")
            print(outputs)
            function_name, function_args, match_info = match

            tool = None
            if hasattr(graphdb_tool, function_name):
                tool = getattr(graphdb_tool, function_name)
            elif hasattr(answer_tool, function_name):
                tool = getattr(answer_tool, function_name)

            if tool:
                tool_result = tool(**function_args)
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
