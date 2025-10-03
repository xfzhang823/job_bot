# Hugging Face Cache Refresh & Lock Pipeline

The `hf_cache_refresh_and_lock_pipeline` ensures all required Hugging Face models
are available **locally** and that the pipeline runs in **strict offline mode**.

## Purpose
- ✅ Pre-download and cache all models your project depends on  
- ✅ Validate cache integrity before running any pipeline stages  
- ✅ Flip environment variables to **force Transformers/BERTScore to use only local files**  
- 🚫 Prevent any accidental internet calls to `huggingface.co`

## How It Works
1. **Optional refresh**  
   - If `refresh_cache=True`: deletes the existing cache, temporarily enables online mode, and re-downloads models.  
   - If `refresh_cache=False`: uses existing cache only.

2. **Cache population**  
   Iterates through a fixed list of required models, calling `AutoModel.from_pretrained` and  
   `AutoTokenizer.from_pretrained` with the correct `cache_dir` to ensure files are downloaded or validated.

3. **Offline enforcement**  
   After caching, sets the following environment variables to guarantee offline execution:
   - `TRANSFORMERS_OFFLINE=1`  
   - `HF_HUB_OFFLINE=1`  
   - `BERT_SCORE_OFFLINE=1`

4. **Logging & validation**  
   - Logs progress and errors per model  
   - Confirms final offline state with `transformers.utils.is_offline_mode()`

## Usage

Call at the **top** of your entrypoint script before importing anything from `transformers` or `bert_score`:

```python
from pipelines.hf_cache_refresh_and_lock_pipeline import run_hf_cache_refresh_and_lock_pipeline

# Use cached models only (no downloads)
run_hf_cache_refresh_and_lock_pipeline(refresh_cache=False)

# Or refresh the cache (requires internet)
# run_hf_cache_refresh_and_lock_pipeline(refresh_cache=True)
