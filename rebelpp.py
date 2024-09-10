import json
from pprint import pformat
import requests
from typing import Literal

import torch
from tap import Tap
from tqdm import tqdm
import wandb

from lmkg.agent import LMKGAgent
from utils import MODELS, LlamaModels, get_model_and_tokenizer


class Arguments(Tap):
    file_path: str
    num_samples: int = 1

    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"
    model: MODELS = LlamaModels.LLAMA_31_8B
    task: str = "contradiction_generation"
    inference_endpoint: str = None
    quantization: Literal["8bit", "4bit"] = None
    max_responses: int = 20

    log_wandb: bool = False

    def configure(self):
        self.add_argument("file_path")


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
        functions="all",
        model=model,
        tokenizer=tokenizer,
        chat_template="llama3-kg",
        inference_endpoint=args.inference_endpoint,
        graphdb_endpoint=args.graphdb_endpoint
    )

    wandb.require("core")
    wandb.init(project='lmkg',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict())
    columns = ["input", "output", "trace"]
    results_table = wandb.Table(columns)

    with open(args.file_path) as f:
        # Iterate over REBEL file
        with tqdm(total=args.num_samples, desc="Processing", mininterval=1) as bar:
            for i, line in enumerate(f):
                bar.update()

                data = json.loads(line)
                passage = data['input']
                triple_ids = data['meta_obj']['non_formatted_wikidata_id_output']
                triple_labels = data['output'][0]['non_formatted_surface_output']

                triples = []
                for t_ids, t_labels in zip(triple_ids, triple_labels):
                    pairs = []
                    for id, label in zip(t_ids, t_labels):
                        pairs.append(f"[{label}:{id}]")

                    triples.append(" ".join(pairs))

                triples = "\n".join(triples)

                task_kwargs = {"passage": passage, "triples": triples}
                try:
                    answer, trace = agent.run(
                        args.task,
                        task_kwargs,
                        args.max_responses,
                        gen_config
                    )
                except requests.exceptions.HTTPError as http_err:
                    answer = "error"
                    trace = ""

                results_table.add_data(pformat(task_kwargs), answer, trace)

                if i + 1 == args.num_samples:
                    break

    wandb.log({"results": results_table})


main(Arguments(explicit_bool=True).parse_args(known_only=True))
