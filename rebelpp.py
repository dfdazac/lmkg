import asyncio
import json
import os
import os.path as osp
import re

import yaml
from langgraph.errors import GraphRecursionError
from tap import Tap
from tqdm import tqdm

from lmkg.agent import LMKGAgent
from lmkg.exceptions import MalformedQueryException
from utils import count_lines, get_timestamp_and_hash


class Arguments(Tap):
    file_path: str = None
    start: int = 0  # Starting line number (0-based)
    end: int = None  # Ending line number (inclusive, 0-based)
    maximum: int = None  # Maximum number of instances to generate

    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"
    task: str = "contradiction_generation"
    functions: list[str] = None
    timeout: int = None
    recursion_limit: int = None

    config_file: str = None


def answer_parser(answer: str) -> tuple[dict, set[str]]:
    """Parse the answer to extract QIDs generated in it. Used to check for hallucination."""
    triple_pattern = re.compile(r'\[([^:\]]+):([PQ]\d+)\] \[([^:\]]+):([PQ]\d+)\] \[([^:\]]+):([PQ]\d+)\]')

    lines = answer.strip().split("\n")
    ids_in_answer = set()

    parsed_triple_ids = []
    parsed_triple_labels = []

    for line in lines:
        match = triple_pattern.fullmatch(line)
        if not match:
            raise ValueError(f"Invalid format detected in line: {line}")

        parsed_triple_labels.append([match.group(1), match.group(3), match.group(5)])
        parsed_triple_ids.append([match.group(2), match.group(4), match.group(6)])
        ids_in_answer.update(parsed_triple_ids[-1])

    answer = {"neg_non_formatted_wikidata_id_output": parsed_triple_ids,
              "neg_non_formatted_surface_output": parsed_triple_labels}

    return answer, ids_in_answer


def main(args: Arguments):
    agent = LMKGAgent(
        functions=args.functions,
        graphdb_endpoint=args.graphdb_endpoint,
        answer_parser=answer_parser,
        timeout=args.timeout,
        recursion_limit=args.recursion_limit
    )

    input_filename = osp.basename(args.file_path)
    output_dir = osp.join(osp.dirname(args.file_path), get_timestamp_and_hash())
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError(f"Directory {output_dir} already exists.")
    output_log = osp.join(output_dir, "log.txt")
    output_file = osp.join(output_dir, f"contradicted-{input_filename}")

    # If end is not specified, process until the end of the file
    if args.end is None:
        total_lines = count_lines(args.file_path)
        args.end = total_lines - 1

    # Calculate the number of lines to process
    lines_to_process = args.end - args.start + 1
    if lines_to_process <= 0:
        raise ValueError(f"Invalid range: start={args.start}, end={args.end}. End must be >= start.")
    if args.maximum is not None and lines_to_process < args.maximum:
        raise ValueError(f"There are {lines_to_process} lines to process "
                         f"but maximum is set to {args.maximum}")

    num_generated = 0
    with open(args.file_path) as f_in, open(output_file, "w") as f_out, open(output_log, "w", buffering=1) as f_log:
        yaml.dump(args.as_dict(), f_log, sort_keys=False, default_flow_style=False)
        # Skip lines until we reach the start line
        for i in range(args.start):
            next(f_in, None)

        total = args.maximum if args.maximum else lines_to_process

        with tqdm(total=total, desc="Generating contradictions", mininterval=1) as bar:
            for line_offset, line in enumerate(f_in):
                current_line_num = args.start + line_offset
                
                # Stop if we've reached the end line
                if current_line_num > args.end:
                    break

                data = json.loads(line)
                passage = data['input']
                triple_ids = data['meta_obj']['non_formatted_wikidata_id_output']
                triple_labels = data['output'][0]['non_formatted_surface_output']
                initial_ids = set()

                triples = []
                for t_ids, t_labels in zip(triple_ids, triple_labels):
                    pairs = []
                    initial_ids.update(t_ids)
                    for id, label in zip(t_ids, t_labels):
                        pairs.append(f"[{label}:{id}]")

                    triples.append(" ".join(pairs))

                triples = "\n".join(triples)

                task_kwargs = {"passage": passage, "triples": triples}
                answer = None
                errors = []
                try:
                    answer, trace = agent.run(
                        args.task,
                        task_kwargs,
                        initial_ids,
                        check_initial_ids=True
                    )
                except GraphRecursionError:
                    errors.append(f"recursion exceeded")
                except asyncio.TimeoutError:
                    errors.append("timed out")
                except KeyError as e:
                    errors.append(f"key error: {e}")
                except MalformedQueryException as e:
                    errors.append("bad query")

                if answer is None:
                    errors.append("no answer")

                sample_log = ",".join(errors) if errors else "ok"
                f_log.write(f"{current_line_num}\t{sample_log}\n")

                if not errors:
                    data['output'].append(answer)
                    f_out.write(f"{json.dumps(data)}\n")
                    num_generated += 1
                    if args.maximum:
                        bar.update()

                if args.maximum is not None and num_generated == args.maximum:
                    break

                if not args.maximum:
                    # No upper bound provided, so we update based on lines in file
                    bar.update()


args = Arguments().parse_args()
if args.config_file:
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    args_dict = args.as_dict()
    args_dict.update(config)
    args = args.from_dict(args_dict)

main(args)
