from typing import Literal
from pprint import pformat

import torch
from tap import Tap
import wandb

from lmkg.agent import LMKGAgent
from utils import MODELS, LlamaModels, get_model_and_tokenizer


class Arguments(Tap):
    task: str
    functions: str

    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"
    model: MODELS = LlamaModels.LLAMA_31_8B
    inference_endpoint: str = None
    quantization: Literal["8bit", "4bit"] = None
    max_responses: int = 20

    log_wandb: bool = False

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

    if args.log_wandb:
        wandb.require("core")
        wandb.init(project='lmkg',
                   mode='online' if args.log_wandb else 'disabled',
                   config=args.as_dict())
        columns = ["input", "output", "trace"]
        table = wandb.Table(columns)

    task_kwargs = dict(arg.lstrip('--').split('=') for arg in args.extra_args)
    answer, trace = agent.run(args.task,
                              task_kwargs,
                              args.max_responses,
                              gen_config)

    if args.log_wandb:
        table.add_data(pformat(task_kwargs),
                       answer,
                       trace)
        wandb.log({"results": table})
    else:
        print(trace)
        print(answer)


main(Arguments(explicit_bool=True).parse_args(known_only=True))
