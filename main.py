from typing import Literal

import torch
from tap import Tap

from lmkg.agent import LMKGAgent
from utils import MODELS, LlamaModels, get_model_and_tokenizer


class Arguments(Tap):
    task: str
    functions: str
    model: MODELS = LlamaModels.LLAMA_31_8B
    inference_endpoint: str = None
    quantization: Literal["8bit", "4bit"] = None
    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"
    max_responses: int = 20

    def configure(self):
        self.add_argument("task")


def main(args: Arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, gen_config = get_model_and_tokenizer(
        args.model,
        args.quantization,
        skip_model=args.inference_endpoint is not None
    )
    if model is not None:
        model = model.to(device)

    agent = LMKGAgent(
        functions=args.functions.split(","),
        model=model,
        tokenizer=tokenizer,
        chat_template="llama3-kg",
        inference_endpoint=args.inference_endpoint,
        graphdb_endpoint=args.graphdb_endpoint
    )

    task_kwargs = dict(arg.lstrip('--').split('=') for arg in args.extra_args)
    answer = agent.run(args.task, task_kwargs, args.max_responses, gen_config)
    print(answer)


main(Arguments().parse_args(known_only=True))
