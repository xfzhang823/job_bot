"""
Filename: hf_cache_refresh_and_lock_pipeline.py

🚀 Purpose:
This pipeline ensures all required Hugging Face models are **refreshed,
cached locally,
and locked in offline mode** before running any pipeline.

*It:
*1. (Optional) Deletes existing cached models (if `refresh_cache=True`).
*2. Caches them locally to avoid network dependencies.
*3. Disables Hugging Face API access to enforce strict offline execution.

🔍 Why Use This?
- Prevents issues caused by **corrupt or incomplete cached models**.
- Ensures **consistent model availability** across all pipelines.
- Avoids **unintended API calls to Hugging Face** (useful for air-gapped or
  production environments).

---
📌 **Usage in `main.py`**
```python
from hf_cache_refresh_and_lock_pipeline import run_hf_cache_refresh_and_lock_pipeline

# ✅ Refresh & lock Hugging Face models before running any pipeline
run_hf_cache_refresh_and_lock_pipeline()

# ✅ Now execute the pipeline
execute_pipeline("2c_async", llm_provider="openai")
"""

"""
Filename: hf_cache_refresh_and_lock_pipeline.py

🚀 Purpose:
This pipeline ensures all required Hugging Face models are **refreshed,
cached locally,
and locked in offline mode** before running any pipeline.
"""

import os
import shutil
import logging
from pathlib import Path
import job_bot.config.logging_config as logging_config
from job_bot.utils.find_project_root import find_project_root

logger = logging.getLogger(__name__)

# ✅ Find and set HF cache directory safely **before importing transformers**
root_dir = find_project_root()
logger.info(f"Root dir: {root_dir}")

HF_CACHE_DIR = (
    root_dir / "hf_cache" if root_dir else Path(r"C:\github\job_bot\hf_cache")
)

# ✅ Ensure the cache directory exists
if not HF_CACHE_DIR.exists():
    logger.warning(f"⚠️ Cache directory does not exist. Creating: {HF_CACHE_DIR}")
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Explicitly set Hugging Face cache variables **before importing transformers**
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)
os.environ["BERT_SCORE_CACHE"] = str(HF_CACHE_DIR / "bert_score")

# 🚫 Ensure offline mode by default **before importing transformers**
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["BERT_SCORE_OFFLINE"] = "1"


def run_hf_cache_refresh_and_lock_pipeline(refresh_cache: bool = False):
    """
    Forces a download of all required models before running any pipeline.

    Args:
        refresh_cache (bool):
            - If `True`, deletes existing cache and forces a fresh download.
            - If `False`, uses cached models without deleting.

    Returns:
        None
    """

    if refresh_cache:
        logger.info("🗑️ Deleting existing Hugging Face cache...")
        try:
            shutil.rmtree(HF_CACHE_DIR)
            logger.info("✅ Cache deleted successfully.")
            HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            logger.warning("⚠️ Cache directory not found. Skipping deletion.")
        except Exception as e:
            logger.error(f"❌ Error deleting cache: {e}")

        # ✅ TEMPORARILY ENABLE ONLINE MODE
        logger.info("🌐 Enabling Hugging Face online mode to refresh cache...")
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"

        # ✅ Import AFTER setting to online mode
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import is_offline_mode

        # ✅ Check environment variables
        transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", "Not Set")
        hf_hub_offline = os.environ.get("HF_HUB_OFFLINE", "Not Set")
        logger.info(f"🔎 TRANSFORMERS_OFFLINE: {transformers_offline}")
        logger.info(f"🔎 HF_HUB_OFFLINE: {hf_hub_offline}")
        logger.info(
            f"🔎 Offline mode status (Before caching models): {is_offline_mode()}"
        )
    else:
        # ✅ Import with default offline mode
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import is_offline_mode

    logger.info("🚀 Starting Hugging Face model cache refresh...")

    model_names = [
        "microsoft/deberta-large-mnli",
        "roberta-large",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/stsb-roberta-base",
        "bert-base-uncased",
    ]

    for model_name in model_names:
        try:
            logger.info(f"🔄 Caching model: {model_name}")
            if refresh_cache:
                AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(HF_CACHE_DIR),
                    force_download=True,
                    local_files_only=False,
                )
                AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(HF_CACHE_DIR),
                    force_download=True,
                    local_files_only=False,
                )
            else:
                AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(HF_CACHE_DIR),
                    local_files_only=True,
                )
                AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(HF_CACHE_DIR),
                    local_files_only=True,
                )
            logger.info(f"✅ Model cached successfully: {model_name}")
        except Exception as e:
            logger.error(f"❌ Error caching {model_name}: {e}")

    logger.info("✅ All models have been refreshed and cached.")

    # ✅ RE-ENFORCE OFFLINE SETTINGS
    logger.info("🛑 Re-enabling Hugging Face offline mode...")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["BERT_SCORE_OFFLINE"] = "1"

    # ✅ FINAL LOGGING
    transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", "Not Set")
    hf_hub_offline = os.environ.get("HF_HUB_OFFLINE", "Not Set")
    logger.info(f"🚫 FINAL TRANSFORMERS_OFFLINE: {transformers_offline}")
    logger.info(f"🚫 FINAL HF_HUB_OFFLINE: {hf_hub_offline}")
    logger.info(f"🚫 FINAL Hugging Face Offline Mode Status: {is_offline_mode()}")


if __name__ == "__main__":
    run_hf_cache_refresh_and_lock_pipeline()
