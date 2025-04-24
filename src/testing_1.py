import os
import logging
import logging_config

logger = logging.getLogger(__name__)

# Ensure offline mode is set
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = r"C:\github\job_bot\hf_cache"
os.environ["BERT_SCORE_CACHE"] = r"C:\github\job_bot\hf_cache\bert_score"

import torch
from bert_score import score
from transformers import AutoTokenizer, AutoModel
import matplotlib


logger = logging.getLogger(__name__)


matplotlib.use("Agg")  # Prevent interactive mode

# Manually load tokenizer and model to ensure it's offline
model_name = "bert-base-uncased"
cache_dir = os.environ["TRANSFORMERS_CACHE"]

tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir=cache_dir, local_files_only=True
)
model = AutoModel.from_pretrained(
    model_name, cache_dir=cache_dir, local_files_only=True
)

# Ensure device is set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Now run BERTScore
cands = ["The cat sat on the mat."]
refs = ["A feline rested on the carpet."]

results = score(
    cands,
    refs,
    model_type=model_name,  # This should now use the manually cached model
    lang="eng",
    rescale_with_baseline=True,
    device=device,
)

# Ensure correct unpacking
if isinstance(results, tuple) and len(results) == 3:
    P, R, F1 = results

logger.info(f"BERTScore Precision: {torch.tensor(P).mean().item():.4f}")
logger.info(f"BERTScore Recall: {torch.tensor(R).mean().item():.4f}")
logger.info(f"BERTScore F1: {torch.tensor(F1).mean().item():.4f}")
