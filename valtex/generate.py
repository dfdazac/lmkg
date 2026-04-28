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
from valtex.utils import count_lines, get_timestamp_and_hash, LLM_COST_MAPPING


class Arguments(Tap):
    file_path: str = None
    start: int = 0  # Starting line number (0-based)
    end: int = None  # Ending line number (inclusive, 0-based)
    max_to_generate: int = None  # Maximum number of instances to generate

    model: str
    base_url: str
    graphdb_endpoint: str = "http://localhost:7200/repositories/wikidata5m"
    task: str = "contradiction_generation"
    functions: list[str] = None
    timeout: int = None
    recursion_limit: int = None

    config_file: str = None


def answer_parser(answer: str) -> tuple[dict, set[str]]:
    """Parse the answer to extract QIDs generated in it. Used to check for hallucination."""
    triple_pattern = re.compile(r'\[([PQ]\d+):([^:\]]+)\] \[([PQ]\d+):([^:\]]+)\] \[([PQ]\d+):([^:\]]+)\]')

    lines = answer.strip().split("\n")
    ids_in_answer = set()

    parsed_triple_ids = []
    parsed_triple_labels = []

    for line in lines:
        match = triple_pattern.fullmatch(line)
        if not match:
            raise ValueError(f"Invalid format detected in line: {line}")

        parsed_triple_ids.append([match.group(1), match.group(3), match.group(5)])
        parsed_triple_labels.append([match.group(2), match.group(4), match.group(6)])
        ids_in_answer.update(parsed_triple_ids[-1])

    answer = {"neg_non_formatted_wikidata_id_output": parsed_triple_ids,
              "neg_non_formatted_surface_output": parsed_triple_labels}

    return answer, ids_in_answer


def judge_answer_parser(answer: str) -> tuple[bool, set[str]]:
    """Parse the answer to extract YES or NO.
    """
    if answer.strip().lower() == "yes":
        judgement = True
    elif answer.strip().lower() == "no":
        judgement = False
    else:
        raise ValueError(f"Invalid answer: {answer}. Must be YES or NO.")

    return judgement, set()


def run_agent_safely(agent: LMKGAgent, agent_name: str, *args, **kwargs):
    """
    Helper to run an agent and capture common exceptions.

    Returns:
        (result, errors)
        - result: whatever `agent.run` returns, or None on error
        - errors: list of string error messages
    """
    try:
        result = agent.run(*args, **kwargs)
        return result, []
    except GraphRecursionError:
        return None, [f"{agent_name}: recursion exceeded"]
    except asyncio.TimeoutError:
        return None, [f"{agent_name}: timed out"]
    except KeyError as e:
        return None, [f"{agent_name}: key error: {e}"]
    except MalformedQueryException:
        return None, [f"{agent_name}: bad query"]


def main(args: Arguments):
    agent = LMKGAgent(
        model=args.model,
        base_url=args.base_url,
        functions=args.functions,
        graphdb_endpoint=args.graphdb_endpoint,
        answer_parser=answer_parser,
        timeout=args.timeout,
        recursion_limit=args.recursion_limit,
        max_label_count=2,
        max_description_length=100
    )
    judge = LMKGAgent(
        model=args.model,
        base_url=args.base_url,
        functions=["get_entity_labels", "get_predicate_description"],
        graphdb_endpoint=args.graphdb_endpoint,
        answer_parser=judge_answer_parser,
        timeout=args.timeout,
        recursion_limit=args.recursion_limit,
        max_label_count=2,
        max_description_length=100
    )
    model_cost = LLM_COST_MAPPING[args.model]

    input_filename = osp.basename(args.file_path)
    output_dir = osp.join(osp.dirname(args.file_path), f"valtex-{get_timestamp_and_hash()}")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError(f"Directory {output_dir} already exists.")
    output_log = osp.join(output_dir, "log.tsv")
    output_file = osp.join(output_dir, f"contradicted-{input_filename}")
    config_file = osp.join(output_dir, "config.yml")

    # Save full configuration for this run
    with open(config_file, "w") as f_config:
        yaml.dump(args.as_dict(), f_config, sort_keys=True, default_flow_style=False)

    # If end is not specified, process until the end of the file
    total_lines = count_lines(args.file_path)
    if args.end is None:
        args.end = total_lines - 1

    # Calculate the number of lines to process
    lines_to_process = args.end - args.start + 1
    if lines_to_process <= 0:
        raise ValueError(f"Invalid range: start={args.start}, end={args.end}. End must be >= start.")

    num_generated = 0
    with open(args.file_path) as f_in, open(output_file, "w") as f_out, open(output_log, "w", buffering=1) as f_log:
        # Write header for tab-separated log file
        f_log.write("id\tstatus\tinput_tokens\toutput_tokens\tcost\n")

        # Track cumulative usage to derive per-sample per-instance values
        prev_total_input_tokens = 0
        prev_total_output_tokens = 0

        # Skip lines until we reach the start line
        for i in range(args.start):
            next(f_in, None)

        with tqdm(total=lines_to_process, desc="Generating", mininterval=1) as bar:
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
                        pairs.append(f"[{id}:{label}]")

                    triples.append(" ".join(pairs))

                triples = "\n".join(triples)

                task_kwargs = {"passage": passage, "triples": triples}
                answer = None
                errors = []
                agent_result, agent_errors = run_agent_safely(
                    agent,
                    "agent",
                    args.task,
                    task_kwargs,
                    initial_ids,
                    check_initial_ids=True,
                )
                if agent_result is not None:
                    answer, reasoning, _ = agent_result
                errors.extend(agent_errors)

                if answer is None:
                    errors.append(f"agent: no answer")

                if not errors:
                    answer_triples = []
                    for labels, identifiers in zip(answer['neg_non_formatted_wikidata_id_output'], answer['neg_non_formatted_surface_output']):
                        answer_triples.append(" ".join([f"[{label}:{id}]" for label, id in zip(labels, identifiers)]))

                    answer_triples = "\n".join(answer_triples)

                    judge_result, judge_errors = run_agent_safely(
                        judge,
                        "judge",
                        task="contradiction_generation_judge",
                        task_kwargs={
                            "text": passage,
                            "triples": triples,
                            "contradicting_triples": answer_triples,
                        },
                    )
                    errors.extend(judge_errors)
                    if not errors:
                        judge_answer, judge_reasoning, _ = judge_result
                        if not judge_answer:
                            errors.append("judge answer is no")
                        else:
                            # Everything worked out! Write the data to the output file.
                            answer['generation_reasoning'] = reasoning
                            answer['judgement_reasoning'] = judge_reasoning
                            data['output'].append(answer)
                            f_out.write(f"{json.dumps(data)}\n")
                            num_generated += 1

                sample_log = ",".join(errors) if errors else "ok"

                # Update progress bar tokens & cost based on agent's accumulated usage
                total_input_tokens, total_output_tokens = agent.get_usage_totals()
                judge_input_tokens, judge_output_tokens = judge.get_usage_totals()
                total_input_tokens += judge_input_tokens
                total_output_tokens += judge_output_tokens

                # Per-instance token deltas
                delta_input_tokens = total_input_tokens - prev_total_input_tokens
                delta_output_tokens = total_output_tokens - prev_total_output_tokens

                # Cumulative total cost
                total_cost = (
                    total_input_tokens * model_cost[0]
                    + total_output_tokens * model_cost[1]
                ) / 1_000_000

                # Per-instance cost from token deltas
                sample_cost = (
                    delta_input_tokens * model_cost[0]
                    + delta_output_tokens * model_cost[1]
                ) / 1_000_000

                # Log metadata (tab-separated): per-instance tokens, per-instance cost
                f_log.write(
                    f"{current_line_num}\t{sample_log}\t{delta_input_tokens}\t{delta_output_tokens}\t{sample_cost:.6f}\n"
                )

                # Update previous totals for next iteration
                prev_total_input_tokens = total_input_tokens
                prev_total_output_tokens = total_output_tokens
                bar.set_postfix(
                    successful=f"{num_generated:,}",
                    input_tokens=f"{total_input_tokens:,}",
                    output_tokens=f"{total_output_tokens:,}",
                    cost=f"${total_cost:.4f}",
                )
                bar.update()

                if args.max_to_generate and num_generated == args.max_to_generate:
                    break


args = Arguments().parse_args()
if args.config_file:
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    args_dict = args.as_dict()
    args_dict.update(config)
    args = args.from_dict(args_dict)

main(args)
