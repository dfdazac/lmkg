from typing import Literal

import torch
from transformers import AwqConfig, BitsAndBytesConfig, pipeline


class LlamaModels:
    LLAMA_31_8B = "Llama-3.1-8B"
    LLAMA_31_70B = "Llama-3.1-70B"
    LLAMA_31_70B_AWQ = "Llama-3.1-70B-AWQ"


# noinspection PyTypeHints
MODELS = Literal[
    LlamaModels.LLAMA_31_8B,
    LlamaModels.LLAMA_31_70B,
    LlamaModels.LLAMA_31_70B_AWQ
]

def get_model(model_name: str, quantization: str = None):
    if quantization is not None:
        if quantization not in ("4bit", "8bit"):
            raise ValueError("Quantization must be either '4bit' or '8bit'")

    if model_name in (LlamaModels.LLAMA_31_8B, LlamaModels.LLAMA_31_70B):
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        pipe = pipeline(
            "text-generation",
            model=f"meta-llama/Meta-{model_name}-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16,
                          "quantization_config": quantization_config},
            device="cuda" if not quantization else None,
        )
    elif model_name == LlamaModels.LLAMA_31_70B_AWQ:
        quantization_config = AwqConfig(
            bits=4
        )
        pipe = pipeline(
            "text-generation",
            model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            model_kwargs={"torch_dtype": torch.float16,
                          "low_cpu_mem_usage": True,
                          "quantization_config": quantization_config},
            device="cuda"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return pipe
