from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv  # pip/uv: python-dotenv

# Toggle via env if you ever want to skip .env loading (e.g., in prod/CI)
SKIP = os.getenv("JOB_BOT_SKIP_DOTENV") == "1"


def load_env_once() -> None:
    """Load .env files and set safe defaults for offline HF usage."""
    if SKIP or os.environ.get("_JOB_BOT_ENV_LOADED") == "1":
        return

    # Prefer project root .env; support .env.local override if present
    root = Path(__file__).resolve().parents[1]  # adjust if your layout differs
    for fname in (".env.local", ".env"):
        f = root / fname
        if f.exists():
            # Do NOT override real environment; let ops/CI take precedence
            load_dotenv(f, override=False)

    # Hardening: offline + deterministic caches (only if not already set)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    hf_home = os.environ.setdefault("HF_HOME", str(root / "hf_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(Path(hf_home) / "sentence-transformers")
    )
    os.environ.setdefault("TORCH_HOME", str(Path(hf_home) / "torch"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Mark loaded to avoid duplicate work if called again
    os.environ["_JOB_BOT_ENV_LOADED"] = "1"
