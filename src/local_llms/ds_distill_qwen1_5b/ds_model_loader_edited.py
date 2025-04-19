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

# Imports
import os
import time
from typing import Optional, Tuple, Union
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# Load the model
load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")
model_name = os.getenv("MODEL_NAME")


class ModelLoader:
    """
    Singleton loader for a Hugging Face causal language model and its tokenizer.

    This class ensures that the model and tokenizer are:
    - Loaded only once and cached across calls
    - Configured with 4-bit quantization for optimized memory usage
    - Automatically selected based on environment variables (`MODEL_NAME`,
    `HUGGING_FACE_TOKEN`)

    Tokenizer Notes:
    - `AutoTokenizer` is used to dynamically return a suitable tokenizer
    (fast or slow).
    - Internally stored as `PreTrainedTokenizerBase` for static type safety.

    Usage:
        >>> tokenizer, model = ModelLoader.load_model()

    Returns:
        Tuple[PreTrainedTokenizerBase, AutoModelForCausalLM]: Cached tokenizer
        and model.
    """

    # class-level variable _models, and they will eventually hold a Hugging Face language
    # model, but are currently unset (None)
    _model: Optional[AutoModelForCausalLM] = None
    _tokenizer: Optional[PreTrainedTokenizerBase] = None

    @classmethod
    def load_model(
        cls,
    ) -> Tuple[
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase],
        AutoModelForCausalLM,
    ]:
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

            if not model_name:
                raise EnvironmentError(
                    "MODEL_NAME not found in .env or environment variables."
                )

            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # * faster but slightly less accurate
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quant_config,
                use_auth_token=token,
            )  # device_map: dynamically balancing between CPU and GPU

            # # Safely load a quantized model on Windows/CPU (no device_map!)
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_name,
            #     quantization_config=quant_config,
            #     use_auth_token=token,
            # )

            cls._model, cls._tokenizer = model, tokenizer

            logger.info("Model and tokenizer loaded and cached.")
            logger.info(f"Model loaded in {round(time.time() - start, 2)}s")

        logger.info("✅ Using cached model from memory")
        return cls._tokenizer, cls._model  # type: ignore
