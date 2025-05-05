# models/model_type.py
from typing import Union
from models.resume_job_description_io_models import (
    NestedResponsibilities,
    Requirements,
    Responsibilities,
    JobPostingUrlsBatch,
    JobPostingsBatch,
    ExtractedRequirementsBatch,
    SimilarityMetrics,
)
from models.llm_response_models import RequirementsResponse

ModelType = Union[
    RequirementsResponse,
    Requirements,
    Responsibilities,
    NestedResponsibilities,
    JobPostingUrlsBatch,
    JobPostingsBatch,
    ExtractedRequirementsBatch,
    SimilarityMetrics,
]
