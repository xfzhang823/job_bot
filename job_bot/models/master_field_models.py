from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# You can import enums like LLMProvider or PipelineStage if you prefer stronger types
class MasterFieldModel(BaseModel):
    # Standard metadata fields
    url: Optional[str] = None
    source_file: Optional[str] = None
    stage: Optional[str] = None
    timestamp: Optional[datetime] = None
    version: Optional[str] = None
    llm_provider: Optional[str] = None
    iteration: Optional[int] = None

    # Job postings / requirements
    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    salary_info: Optional[str] = None
    posted_date: Optional[str] = None
    content: Optional[str] = None  # Stored as JSON string

    # Requirements (flattened or extracted)
    requirement_key: Optional[str] = None
    requirement: Optional[str] = None
    requirement_category: Optional[str] = None
    requirement_idx: Optional[int] = None
    requirement_category_idx: Optional[int] = None

    # Responsibilities
    responsibility_key: Optional[str] = None
    responsibility: Optional[str] = None
    # optimized_text: Optional[str] = None # commented out (standardize on responsibility col name)
    pruned_by: Optional[str] = None

    # Similarity metrics
    similarity_score: Optional[float] = None
    entailment_score: Optional[float] = None
    composite_score: Optional[float] = None
    pca_score: Optional[float] = None

    # Human review
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    trimmed_by: Optional[str] = None
