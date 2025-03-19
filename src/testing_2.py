from transformers.utils import logging
from transformers import AutoModel, AutoTokenizer
import os

# Enable debug logging
logging.get_logger("transformers").setLevel("DEBUG")

# ✅ Explicitly print current Hugging Face cache settings
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))

# ✅ Load a model and check where it's caching
model_name = "bert-base-uncased"

model = AutoModel.from_pretrained(
    model_name, cache_dir=os.environ.get("TRANSFORMERS_CACHE")
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir=os.environ.get("TRANSFORMERS_CACHE")
)

print("✅ Model loaded successfully.")

import os
from transformers.utils import is_offline_mode

print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
print(f"Offline mode status: {is_offline_mode()}")
