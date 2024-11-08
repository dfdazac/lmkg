from typing import Literal

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)


def get_model_and_tokenizer(model_id: str,
                            quantization: str = None,
                            skip_model: bool = False):
    """Load a model and a tokenizer given their Hugging Face IDs. The model can
    be skipped if needed, e.g. when running it over Text Generation Inference
    and only the tokenizer is needed.

    Args:
        model_id: The Hugging Face Hub ID of the model to load.
        quantization: The quantization kind to use, one of "4bit", "8bit"
        skip_model: Whether to skip the model loading or not.
    """
    if quantization is not None:
        if quantization not in ("4bit", "8bit"):
            raise ValueError("Quantization must be either '4bit' or '8bit'")

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

    model = None
    if not skip_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=quantization_config,
            torch_dtype=torch.bfloat16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    gen_config = GenerationConfig.from_pretrained(model_id)

    return model, tokenizer, gen_config
