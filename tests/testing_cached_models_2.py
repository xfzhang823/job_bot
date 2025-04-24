from transformers import AutoModel, AutoTokenizer
from utils.find_project_root import find_project_root

project_dir = find_project_root()

if project_dir:
    hf_cache_dir = project_dir / "hf_cache"
else:
    raise RuntimeError("Project root not found!")


models_to_test = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "bert-base-uncased",
    "sentence-transformers/stsb-roberta-base",
    "microsoft/deberta-large-mnli",
]

for model_name in models_to_test:
    print(f"\n🔍 Testing if '{model_name}' loads `roberta-large` internally...\n")

    try:
        model = AutoModel.from_pretrained(
            model_name, cache_dir=hf_cache_dir, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=hf_cache_dir, local_files_only=True
        )
        print(f"✅ '{model_name}' loaded successfully from cache!\n")
    except Exception as e:
        print(
            f"❌ '{model_name}' is missing files or calling Hugging Face!\nError: {e}\n"
        )
