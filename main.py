from pprint import pformat

from tap import Tap
import wandb

from lmkg.agent import LMKGAgent


class Arguments(Tap):
    task: str = "entity_linking"
    functions: str = "search_entities"

    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"
    max_responses: int = 20

    log_wandb: bool = False

    def configure(self):
        self.add_argument("task")


def main(args: Arguments):
    agent = LMKGAgent(
        functions=args.functions.split(","),
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
                              args.max_responses)

    if args.log_wandb:
        table.add_data(pformat(task_kwargs),
                       answer,
                       trace)
        wandb.log({"results": table})
    else:
        print(trace)
        print(answer)


main(Arguments(explicit_bool=True).parse_args(known_only=True))
