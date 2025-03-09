"""
Module: model_loader.py

Utility module for managing multiple SentenceTransformer and Transformer-based
models efficiently.

This module provides globally cached functions to ensure models are loaded
only once, preventing redundant API calls and improving performance.

Pretrained Models Used:
-----------------------
- `sentence-transformers/all-MiniLM-L6-v2` (SBERT)
- `bert-base-uncased` (BERT)
- `stsb-roberta-base` (STS)
- `microsoft/deberta-large-mnli` (DeBERTa)
- `spaCy en_core_web_md` (for NLP processing)

Functions:
----------
- `get_model(model_name)`: Loads and caches the specified Hugging Face model.
- `get_spacy_model(model_name)`: Loads and caches the specified spaCy model.
"""

# Dependencies
from pathlib import Path
import logging
import traceback
from sentence_transformers import SentenceTransformer
from transformers import (
    BertTokenizer,
    BertModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
)
import spacy
import subprocess
from utils.generic_utils import find_project_root

# Logger
# Setup logging to capture DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,  # ✅ Ensures DEBUG-level messages are logged
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

logger.debug("🚀 DEBUG: Logging setup is working!")

# Attempt to find the project root
PROJECT_ROOT = find_project_root()

# Handle case where project root is not found
if PROJECT_ROOT is None:
    raise RuntimeError(
        "❌ Project root not found. Ensure you have a valid root directory."
    )

# Define cache directories (but don't create them yet)
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
SP_CACHE_DIR = PROJECT_ROOT / "sp_cache"

# Cache dictionary for loaded models
MODEL_CACHE = {}

# # List of models to cache
# PRETRAINED_MODELS = {
#     "sbert": "sentence-transformers/all-MiniLM-L6-v2",  # Corrected full model name
#     "bert": "bert-base-uncased",
#     "sts": "sentence-transformers/stsb-roberta-base",  # Corrected full model name
#     "deberta": "microsoft/deberta-large-mnli",
#     "bert_score": "bert-base-uncased",
# }


def debug_model_loading(model_name):
    """
    Debug when a model is being loaded.
    Prints a full traceback to identify where `roberta-large` is being called.
    """
    logger.debug(f"🛠 DEBUG: Checking model loading: {model_name}")
    if "roberta-large" in model_name:
        logger.warning(f"⚠️ WARNING: `roberta-large` is being loaded!")
        traceback.print_stack()


def get_hf_model(model_name: str):
    """
    Load and return a cached transformer model.

    Args:
        model_name (str): Full Hugging Face model name.

    Returns:
        Transformer model instance.
    """
    global MODEL_CACHE

    if model_name not in MODEL_CACHE:
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure cache exists
        logger.info(f"📥 Loading model '{model_name}' from cache...")

        try:
            if "sentence-transformers" in model_name:  # SBERT or STS models
                MODEL_CACHE[model_name] = SentenceTransformer(
                    model_name, cache_folder=str(HF_CACHE_DIR), local_files_only=True
                )
            elif "deberta" in model_name:  # DeBERTa entailment model
                MODEL_CACHE[model_name] = {
                    "tokenizer": AutoTokenizer.from_pretrained(
                        model_name, cache_dir=str(HF_CACHE_DIR), local_files_only=True
                    ),
                    "model": AutoModelForSequenceClassification.from_pretrained(
                        model_name, cache_dir=str(HF_CACHE_DIR), local_files_only=True
                    ),
                }
            else:  # Default for BERT-based models
                MODEL_CACHE[model_name] = {
                    "tokenizer": AutoTokenizer.from_pretrained(
                        model_name, cache_dir=str(HF_CACHE_DIR), local_files_only=True
                    ),
                    "model": AutoModel.from_pretrained(
                        model_name, cache_dir=str(HF_CACHE_DIR), local_files_only=True
                    ),
                }

            logger.info(f"✅ Model '{model_name}' loaded successfully from cache.")

        except Exception as e:
            logger.exception(f"❌ Error loading '{model_name}'")
            raise e

    return MODEL_CACHE[model_name]


def get_spacy_model(model_name="en_core_web_md"):
    """
    Load and cache a spaCy NLP model.
    """
    global MODEL_CACHE

    if model_name not in MODEL_CACHE:
        model_path = SP_CACHE_DIR / model_name  # Store in `sp_cache`

        # Check if model is already installed
        try:
            nlp = spacy.load(model_name)
            MODEL_CACHE[model_name] = nlp  # Cache it
        except OSError:  # Model not found
            logger.info(f"📥 Downloading spaCy model {model_name} to {model_path}...")

            # Use subprocess to call `python -m spacy download`
            try:
                subprocess.run(
                    ["python", "-m", "spacy", "download", model_name], check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"❌ Failed to download spaCy model {model_name}: {e}"
                )

            # Load model after downloading
            MODEL_CACHE[model_name] = spacy.load(model_name)

    return MODEL_CACHE[model_name]
