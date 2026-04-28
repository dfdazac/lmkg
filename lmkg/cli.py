from pprint import pformat

from tap import Tap

from lmkg.agent import LMKGAgent


class Arguments(Tap):
    model: str
    base_url: str
    task: str = "entity_linking"
    functions: str = "search_entities"

    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"
    recursion_limit: int = 20

    def configure(self):
        self.add_argument("task")


def main(args: Arguments):
    agent = LMKGAgent(
        model=args.model,
        base_url=args.base_url,
        functions=args.functions.split(","),
        graphdb_endpoint=args.graphdb_endpoint,
        recursion_limit=args.recursion_limit
    )

    task_kwargs = dict(arg.lstrip('--').split('=') for arg in args.extra_args)
    answer, reasoning, trace = agent.run(args.task, task_kwargs)

    print(answer)


if __name__ == "__main__":
    args = Arguments(explicit_bool=True).parse_args(known_only=True)
    main(args)
