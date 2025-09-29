# job_bot/fsm/load_stage_config.py
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import importlib, os, re, yaml
from pathlib import Path


# ---------- Pydantic models ----------
class TransitionCfg(BaseModel):
    stage: str
    status: str


class StageCfg(BaseModel):
    runner: str  # "module.sub:func"
    reads_from: str  # DSL or SQL snippet
    writes_to: List[str] = Field(default_factory=list)
    concurrency: int = 4
    batch_size: int = 1
    on_success: Optional[TransitionCfg] = None
    on_error: Optional[TransitionCfg] = None


class StagesFile(BaseModel):
    defaults: StageCfg = StageCfg(runner="__invalid__:__invalid__", reads_from="")
    stages: Dict[str, StageCfg]

    def resolved(self) -> Dict[str, StageCfg]:
        """Apply defaults to stages and return a plain dict."""
        out: Dict[str, StageCfg] = {}
        for k, v in self.stages.items():
            # inherit defaults for missing fields
            merged = self.defaults.model_copy(update=v.model_dump(exclude_unset=True))
            out[k] = merged
        return out


# ---------- Loader ----------
_env_pat = re.compile(r"\$\{(\w+)\}")


def _expand_env(text: str) -> str:
    return _env_pat.sub(lambda m: os.getenv(m.group(1), ""), text)


def load_stages_config(path: str | Path) -> Dict[str, StageCfg]:
    p = Path(path)
    data = yaml.safe_load(_expand_env(p.read_text())) or {}
    cfg = StagesFile.model_validate(data)
    return cfg.resolved()


# ---------- Optional: runner importer ----------
def import_runner(path: str):
    """Import a runner given 'module.sub:func'."""
    try:
        mod, func = path.split(":")
        return getattr(importlib.import_module(mod), func)
    except Exception as e:
        raise ImportError(f"Failed to import runner '{path}': {e}") from e
