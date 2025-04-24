from transformers import AutoModel, AutoTokenizer

# ✅ Ensure cache directory exists
import os

HF_CACHE_DIR = r"C:\github\job_bot\hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# ✅ List of models to download
MODEL_NAMES = [
    "microsoft/deberta-large-mnli",
    "roberta-large",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/stsb-roberta-base",
    "bert-base-uncased",
    "roberta-large-mnli",
]

# ✅ Download and store in the correct cache directory
for model_name in MODEL_NAMES:
    print(f"🔄 Downloading: {model_name}")
    AutoModel.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    print(f"✅ Downloaded and cached: {model_name}")
