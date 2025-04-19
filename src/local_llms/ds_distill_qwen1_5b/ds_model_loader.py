"""
/local_llms/ds_distill_qwen1_5b/ds_model_loader.py

Tokenizer Types Summary (Hugging Face 🤗 Transformers)

AutoTokenizer:
- Factory method, not a class/type.
- Dynamically returns either a "fast" or "slow" tokenizer depending on the model and availability.

PreTrainedTokenizerFast:
- Backed by Hugging Face's Rust-based `tokenizers` library.
- Extremely fast and used by default for most modern models.
- Class: `PreTrainedTokenizerFast`

PreTrainedTokenizer:
- Legacy Python-only tokenizer.
- Slower, used when a fast tokenizer isn't available for a specific model.
- Class: `PreTrainedTokenizer`

PreTrainedTokenizerBase:
- Abstract base class for both fast and slow tokenizers.
- ✅ Recommended type for safe static typing (`Optional[PreTrainedTokenizerBase]`).

Type Hint Recommendation:
- Use `PreTrainedTokenizerBase` for tokenizer attributes or method returns.
- Do NOT annotate with `AutoTokenizer` — it's a factory, not a real class.

Example:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

Tokenizer Types Summary (Hugging Face 🤗 Transformers)

AutoTokenizer:
- Factory method, not a class/type.
- Dynamically returns either a "fast" or "slow" tokenizer depending on the model and availability.

PreTrainedTokenizerFast:
- Backed by Hugging Face's Rust-based `tokenizers` library.
- Extremely fast and used by default for most modern models.
- Class: `PreTrainedTokenizerFast`

PreTrainedTokenizer:
- Legacy Python-only tokenizer.
- Slower, used when a fast tokenizer isn't available for a specific model.
- Class: `PreTrainedTokenizer`

PreTrainedTokenizerBase:
- Abstract base class for both fast and slow tokenizers.
- ✅ Recommended type for safe static typing (`Optional[PreTrainedTokenizerBase]`).

Type Hint Recommendation:
- Use `PreTrainedTokenizerBase` for tokenizer attributes or method returns.
- Do NOT annotate with `AutoTokenizer` — it's a factory, not a real class.

Example:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
"""

import os
import time
import torch
import logging
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton loader for the model and tokenizer.
    Ensures the model is only loaded once and reused across calls.
    """

    # class-level variable _models, and they will eventually hold a Hugging Face language
    # model, but are currently unset (None)
    _model: AutoModelForCausalLM = None
    _tokenizer: AutoTokenizer = None

    @classmethod
    def load_model(cls) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load and return the tokenizer and model, reusing them if already loaded.

        Returns:
            Tuple[AutoTokenizer, AutoModelForCausalLM]: Tokenizer and model instances.
        """
        if (
            cls._model is None or cls._tokenizer is None
        ):  # * This keeps the model "warm"
            logger.info("🔄 Loading model from disk...")
            start = time.time()

            load_dotenv()
            model_name = os.getenv("MODEL_NAME")

            if not model_name:
                raise EnvironmentError(
                    "MODEL_NAME not found in .env or environment variables."
                )

            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # * faster but slightly less accurate)
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,  # changed from "auto" to None
                quantization_config=quant_config,
                low_cpu_mem_usage=False,  # add this
                torch_dtype=torch.float16,
                trust_remote_code=True,  # add this
            )  # device_map: dynamically balancing between CPU and GPU

            cls._model, cls._tokenizer = model, tokenizer

            logger.info("Model and tokenizer loaded and cached.")
            logger.info(f"Model loaded in {round(time.time() - start, 2)}s")

        logger.info("✅ Using cached model from memory")
        return cls._tokenizer, cls._model
