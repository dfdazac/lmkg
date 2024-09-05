from pprint import pprint
from typing import Any, Optional

import torch
from huggingface_hub import InferenceClient
from transformers import PreTrainedModel, PreTrainedTokenizer

from .tools import AnswerStoreTool, GraphDBTool
from .utils import build_task_input, get_chat_template, run_if_callable


class LMKGAgent:
    def __init__(self,
                 functions: list[str],
                 model: Optional[PreTrainedModel],
                 tokenizer: PreTrainedTokenizer,
                 chat_template: str,
                 inference_endpoint: Optional[str],
                 graphdb_endpoint: str):
        self.model = model
        self.tokenizer = tokenizer
        self.inference_client = None
        if inference_endpoint:
            self.client = InferenceClient(inference_endpoint)
        self.graphdb_endpoint = graphdb_endpoint

        self.chat_template = get_chat_template(chat_template)
        self.graphdb = GraphDBTool(functions, graphdb_endpoint)

        self.messages = []

    def _append_message(self, role: str, content: Any):
        self.messages.append({
            "role": role,
            "content": content
        })

    def __call__(self,
                 task: str,
                 task_kwargs,
                 gen_config) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        answer_store = AnswerStoreTool()

        # Build system prompt, tool definition prompt, and task description
        task_prompt = build_task_input(task, task_kwargs)
        self.messages = []
        self._append_message("user", task_prompt)

        done = False
        printed_system = False
        # Generation loop
        while not done:
            inputs = self.tokenizer.apply_chat_template(
                self.messages,
                tools=self.graphdb.tools_json + answer_store.tools_json,
                chat_template=self.chat_template,
                tokenize=self.inference_client is None,
                add_generation_prompt=True,
                return_dict=self.inference_client is None,
                return_tensors="pt"
            )

            if self.inference_client:
                # Generating from TGI endpoint
                outputs = self.client.text_generation(
                    inputs,
                    return_full_text=False,
                    max_new_tokens=512,
                    do_sample=gen_config.do_sample,
                    temperature=gen_config.temperature,
                    top_k=gen_config.top_k,
                    top_p=gen_config.top_p,
                    details=False
                )
            else:
                # Generating from local model
                inputs = inputs.to(device)
                input_length = inputs['input_ids'].shape[-1]
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    pad_token_id=self.tokenizer.eos_token_id,
                    generation_config=gen_config)

                outputs = self.tokenizer.decode(outputs[0][input_length:])

            if not printed_system:
                if not self.inference_client:
                    inputs = self.tokenizer.decode(inputs['input_ids'][0])
                print(inputs)
                printed_system = True

            self._append_message("assistant", outputs)
            print(outputs)

            tool_result, match_info = run_if_callable(
                outputs,
                tools=[self.graphdb, answer_store]
            )
            if tool_result:
                if match_info:
                    pprint(match_info)
                    self._append_message("ipython",
                                         {"output": match_info})
                pprint(tool_result)
                self._append_message("ipython",
                                     {"output": tool_result})
            elif answer_store.answer is None:
                self._append_message("user",
                                     "You forgot to submit the answer!")
            else:
                done = True

        if not answer_store.answer:
            raise RuntimeError("Generation loop finished but no answer found")

        return answer_store.answer
