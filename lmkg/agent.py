import asyncio
import warnings
from typing import Any, Callable, Optional

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.prebuilt import create_react_agent, ToolNode
import pydantic

from .tools import AnswerStoreTool, GraphDBTool
from .utils import build_task_input


class LMKGAgent:
    """
    An agent designed to interact with a pre-trained language model and a
    knowledge graph database (GraphDB). It facilitates the generation of
    text-based responses using a language model and integrates with tools to
    process and retrieve information from a knowledge graph.

    Args:
        model (str): The language model name to use.
        base_url (str): The OpenAI-compatible API base URL for model requests.
        functions (list[str]): A list of function names to use with the graph database,
            or the string "all" to use all available functions.
        graphdb_endpoint (str): The URL endpoint for accessing the graph database.
        timeout (int, optional): The maximum time in seconds to allow for agent execution.
        recursion_limit (int, optional): The maximum recursion depth for the agent's execution.
        max_label_count (int, optional): The maximum number of labels to return for an entity.
        max_description_length (int, optional): The maximum length of the description to return for an entity.
    """
    def __init__(self,
                 model: str,
                 base_url: str,
                 functions: list[str],
                 graphdb_endpoint: str,
                 answer_parser: Callable[[str], tuple[Any, set[str]]] = None,
                 timeout: int = None,
                 recursion_limit: int = None,
                 max_label_count: int = 3,
                 max_description_length: int = 100):
        self.graphdb_endpoint = graphdb_endpoint
        self.graphdb = GraphDBTool(graphdb_endpoint, functions, max_label_count, max_description_length)
        self.answer_store = AnswerStoreTool(self.graphdb, answer_parser)
        tool_list = self.graphdb.tools + self.answer_store.tools

        model = ChatOpenAI(
            model=model,
            max_retries=2,
            base_url=base_url,
        )
        model = model.bind_tools(tool_list, parallel_tool_calls=False)
        tools = ToolNode(tool_list, handle_tool_errors=(pydantic.ValidationError,))
        self.agent = create_react_agent(model, tools)

        self.timeout = timeout
        self.recursion_limit = recursion_limit

        # Accumulated usage across all invocations of this agent instance
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _accumulate_usage(self, usage_metadata: dict) -> None:
        """
        Accumulate token usage from a single invocation into the running totals.
        """
        if not usage_metadata:
            return
        for usage in usage_metadata.values():
            # We only care about the high-level input/output token counts
            self.total_input_tokens += usage.get("input_tokens", 0)
            self.total_output_tokens += usage.get("output_tokens", 0)

    def get_usage_totals(self) -> tuple[int, int]:
        """
        Get the accumulated input and output token counts for this agent.
        """
        return self.total_input_tokens, self.total_output_tokens

    async def _invoke_agent(self, agent, prompt):
        # Suppress the beta warning raised when instantiating UsageMetadataCallbackHandler
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*UsageMetadataCallbackHandler.*",
                category=Warning,
            )
            usage_callback = UsageMetadataCallbackHandler()
        try:
            response = await asyncio.wait_for(
                agent.ainvoke(
                    input={"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "recursion_limit": self.recursion_limit,
                        "callbacks": [usage_callback],
                    },
                ),
                timeout=self.timeout,
            )
            return response
        finally:
            self._accumulate_usage(usage_callback.usage_metadata)

    def run(self,
            task: str,
            task_kwargs: dict[str, str],
            initial_ids: set[str] = None,
            check_initial_ids: bool = False,
            ) -> tuple[Optional[str], str, str]:
        """
        Executes the agent's main task loop. It generates a response based on
        the task and its arguments, iterating over the conversation until a
        final answer is found.

        Args:
            task: The name or description of the task the agent should perform.
            task_kwargs: A dictionary of keyword arguments to further specify
                the task parameters.
            initial_ids: Initial KG identifiers to allow during hallucination detection
            check_initial_ids: Whether to check if the initial IDs are valid

        Returns:
            The final answer generated by the agent after iterating through the
                conversation and tool results.
        """
        if not self.graphdb.is_alive():
            raise ConnectionError("GraphDB is not running!")
        
        if check_initial_ids:
            for id in initial_ids:
                self.graphdb.check_id_in_graph(id)

        self.answer_store.initialize(initial_ids)
        self.graphdb.clear_session_ids()

        task_prompt = build_task_input(task, task_kwargs)

        response = asyncio.run(self._invoke_agent(self.agent, task_prompt))

        return self.answer_store.answer, self.answer_store.reasoning, response
