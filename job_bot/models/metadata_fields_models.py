from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from db_io.pipeline_enums import PipelineStage, Version, LLMProvider


class MetadataField(str, Enum):
    SOURCE_FILE = "source_file"
    STAGE = PipelineStage.value
    TIMESTAMP = "timestamp"
    VERSION = Version
    LLM_PROVIDER = "llm_provider"
    ITERATION = "iteration"


class CommonMetadata(BaseModel):
    source_file: Optional[str] = None
    stage: Optional[PipelineStage] = None
    timestamp: Optional[datetime] = None
    version: Optional[str] = None
    llm_provider: Optional[LLMProvider] = None
    iteration: Optional[int] = None
