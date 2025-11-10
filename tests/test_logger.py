"""
tests/test_logger_package.py

Smoke test to verify that `logging.getLogger(__name__)` messages
propagate to the root logger and get written to a rotating file in ./logs.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]  # one level up from tests/
sys.path.insert(0, str(ROOT))


import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import getpass


# ---------- logging init (self-contained) ----------


def _find_project_root(marker: str = ".git") -> Path:
    """Search upward from this file's directory and CWD for a marker like '.git'."""
    candidates = []
    here = Path(__file__).resolve()
    candidates.append(here.parent)
    candidates.extend(here.parents)
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)
    for c in candidates:
        if (c / marker).exists():
            return c
    # fallback: cwd
    return Path.cwd().resolve()


def _make_logfile(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    username = getpass.getuser()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return logs_dir / f"{username}_{ts}_app.log"


def init_logging() -> Path:
    """Reset root handlers and attach console + rotating file handlers."""
    root = logging.getLogger()
    # Remove any pre-existing handlers (VS Code/uv may add some)
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    project_root = _find_project_root(".git")
    logfile = _make_logfile(project_root / "logs")

    file_h = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=2_000_000, backupCount=3
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(fmt)

    root.addHandler(file_h)
    root.addHandler(console_h)

    root.debug("✅ Logging initialized")
    root.info(f"Logs → {logfile}")
    return logfile


# ---------- a module-like logger using __name__ ----------

logger = logging.getLogger(__name__)


def do_module_logging():
    logger.debug("DBG from %s", __name__)
    logger.info("INFO from %s", __name__)
    logger.warning("WARN from %s", __name__)
    logger.error("ERROR from %s", __name__)


# ---------- run ----------


def main() -> int:
    logfile = init_logging()

    # Show attached handlers
    root = logging.getLogger()
    print("Attached handlers:", [type(h).__name__ for h in root.handlers])

    # Prove propagation works
    do_module_logging()

    # Force flush
    for h in root.handlers:
        try:
            h.flush()
        except Exception:
            pass

    # Read tail of the file
    try:
        lines = logfile.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        print("❌ Log file not found (did init_logging run before logging?)")
        return 2

    print("\n--- tail of log file ---")
    for line in lines[-10:]:
        print(line)
    print("--- end tail ---")
    print(f"\nOK ✅ log file at: {logfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
