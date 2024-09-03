import json
from pprint import pprint
from typing import Literal

import torch
from huggingface_hub import InferenceClient
from tap import Tap

from lmkg.tools import AnswerStoreTool, GraphDBTool
from lmkg.utils import build_task_input, get_chat_template, run_if_callable
from utils import MODELS, LlamaModels, get_model_and_tokenizer


class Arguments(Tap):
    file_path: str
    model: MODELS = LlamaModels.LLAMA_31_8B
    inference_client: str = None
    quantization: Literal["8bit", "4bit"] = None
    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"

    def configure(self):
        self.add_argument("file_path")


def main(args: Arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, gen_config = get_model_and_tokenizer(
        args.model,
        args.quantization,
        skip_model=args.inference_client is not None
    )
    if model is not None:
        model = model.to(device)
    client = None
    if args.inference_client:
        client = InferenceClient(args.inference_client)

    chat_template = get_chat_template("llama3-kg")
    graphdb_tool = GraphDBTool(args.graphdb_endpoint)
    answer_tool = AnswerStoreTool()

    with open(args.file_path) as f:
        for line in f:
            data = json.loads(line)
            passage = data['input']
            triples_ids = data['meta_obj']['non_formatted_wikidata_id_output']
            triples_labels = data['output'][0]['non_formatted_surface_output']

            triples = []
            for t_ids, t_labels in zip(triples_ids, triples_labels):
                pairs = []
                for id, label in zip(t_ids, t_labels):
                    pairs.append(f"<{label}:{id}>")

                triples.append(" ".join(pairs))

            triples = "\n".join(triples)

            messages = []
            printed_system = False
            prompt = build_task_input("contradiction_generation",
                                     dict(passage=passage,
                                          triples=triples)
                                     )
            messages.append({"role": "user",
                             "content": prompt})

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

                if not client:
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
                        max_new_tokens=512,
                        do_sample=gen_config.do_sample,
                        temperature=gen_config.temperature,
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
                print(outputs)

                tool_result, match_info = run_if_callable(outputs,
                                                          tools=[graphdb_tool,
                                                                 answer_tool])
                if tool_result:
                    if match_info:
                        pprint(match_info)
                        messages.append({"role": "ipython",
                                         "content": {"output": match_info}})

                    pprint(tool_result)
                    messages.append({"role": "ipython",
                                     "content": {"output": tool_result}})
                else:
                    done = True

            print("*" * 50)
            print(answer_tool.answer)

            break


main(Arguments().parse_args(known_only=True))
