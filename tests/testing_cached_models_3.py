import sys

sys.path.insert(0, "../src")

from transformers import AutoModel, AutoTokenizer

import logging
from pathlib import Path


from utils.find_project_root import (
    find_project_root,
)

# Define cache directory
project_dir = find_project_root()

if project_dir:
    HF_CACHE_DIR = project_dir / "hf_cache"
else:
    raise RuntimeError("Project root not found!")

# Enable logging to check if Hugging Face API is called
logging.basicConfig(level=logging.DEBUG)

print("🚀 Testing if 'bert-base-uncased' loads from cache...")

# Try loading the model & tokenizer from cache
try:
    model = AutoModel.from_pretrained(
        "bert-base-uncased", cache_dir=HF_CACHE_DIR, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", cache_dir=HF_CACHE_DIR, local_files_only=True
    )
    print("✅ 'bert-base-uncased' loaded successfully from cache!")
except Exception as e:
    print(f"❌ Error loading model from cache: {e}")
