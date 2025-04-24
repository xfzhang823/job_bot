import os
import logging
from transformers.utils import is_offline_mode
from transformers import AutoModel, AutoTokenizer
from utils.model_loader import get_hf_model, get_spacy_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if Hugging Face is in offline mode
logger.info(
    f"Offline mode: {is_offline_mode()}"
)  # Should print False if online mode is enabled

# Define model names
model_name = "bert-base-uncased"
hf_cache_dir = "./hf_cache"

# Check if cache directory exists
if not os.path.exists(hf_cache_dir):
    logger.warning(f"⚠️ Hugging Face cache directory '{hf_cache_dir}' does not exist.")

# List cache contents
logger.info(
    f"📂 Contents of {hf_cache_dir}: {os.listdir(hf_cache_dir) if os.path.exists(hf_cache_dir) else 'Directory not found'}"
)

# Test Hugging Face model loading from cache
try:
    model = AutoModel.from_pretrained(
        model_name, local_files_only=True, cache_dir=hf_cache_dir
    )
    logger.info(f"✅ Model '{model_name}' loaded from cache!")
except Exception as e:
    logger.error(f"❌ Model '{model_name}' not found in cache: {e}")

# Check if models are correctly mapped in PRETRAINED_MODELS
hf_models = ["sbert", "bert", "deberta"]

for model_key in hf_models:
    try:
        model = get_hf_model(model_key)
        logger.info(f"✅ HF Model '{model_key}' loaded successfully from cache!")
    except KeyError as e:
        logger.error(f"❌ Model '{model_key}' not found in PRETRAINED_MODELS: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to load model '{model_key}': {e}")

# Test spaCy model loading
try:
    nlp = get_spacy_model("en_core_web_md")
    logger.info("✅ spaCy model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load spaCy model: {e}")

logger.info("✅ All tests completed!")
