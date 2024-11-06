"""
File: base_models.py
Last Updated on:

pydantic models for validate LLM responses
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ValidationError, constr, Field

import pandas as pd
import logging
import logging_config


# Set up logger
logger = logging.getLogger(__name__)


# Base model to serve as a foundation for multiple response types
class BaseResponseModel(BaseModel):
    status: str = "success"  # General status field for all responses
    message: Optional[str] = None  # Optional message or feedback

    class Config:
        arbitrary_types_allowed = True  # Allow non-standard types like DataFrame


# Model for handling text-based responses
class TextResponse(BaseResponseModel):
    content: str  # Holds plain text content

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Text response processed.",
                "content": "This is the plain text content.",
            }
        }


# Generalized JSON Response Model (New Version)
class JSONResponse(BaseModel):
    """Basic and generic model for JSON response"""

    data: Union[
        Dict[str, Any], List[Dict[str, Any]]
    ]  # Allow both dict and list of dicts

    class Config:
        arbitrary_types_allowed = True


# Old version:
# class JSONResponse(BaseResponseModel):
#     optimized_text: Optional[str] = None  # For any optimized text-based content
#     data: Optional[dict] = None  # General data field for various JSON responses

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "status": "success",
#                 "message": "JSON response processed.",
#                 "optimized_text": "This is the optimized version of your text.",
#                 # keep optimized_text in the base model b/c other models such as summarization, etc. may need it
#                 "data": {"key_1": "value_1", "key_2": "value_2"},
#             }
#         }


# Model for handling tabular responses using pandas DataFrame
class TabularResponse(BaseResponseModel):
    """Generic modle for tabular LLM response"""

    data: pd.DataFrame  # Tabular data, returned as a pandas DataFrame

    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrame as a valid type
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Tabular data processed.",
                "data": "Pandas DataFrame object",
            }
        }


# Model for handling code responses (e.g., snippets of code)
class CodeResponse(BaseResponseModel):
    """Generic modle for tabular LLM response"""

    code: str  # Holds the code as a string

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Code response processed.",
                "code": "print('Hello, world!')",
            }
        }


# Example job-specific response inheriting from JSONResponse (for job processing pipelines)
class JobSiteResponseModel(JSONResponse):
    """Specific model to validate job posting site parsing"""

    url: Optional[str] = Field(None, description="Job posting URL")
    job_title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    salary_info: Optional[str] = Field(None, description="Salary information")
    posted_date: Optional[str] = Field(None, description="Job posting date")
    content: Optional[dict] = Field(
        None,
        description="Dictionary containing job description, responsibilities, and qualifications",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Job data extracted successfully.",
                "url": "https://searchjobs.libertymutualgroup.com/careers/job/618499888480?microsite=libertymutual.com&domain=libertymutual.com&utm_source=Job+Board&utm_campaign=LinkedIn+Jobs&extcmp=bof-paid-text-lkin-aljb",
                "job_title": "Software Engineer",
                "company": "Tech Corp",
                "location": "New York, NY",
                "salary_info": "$100,000 - $120,000 per year",
                "posted_date": "2024-10-01",
                "content": {
                    "description": "Design and build new features...",
                    "responsibilities": "Lead team meetings...",
                    "qualifications": "Bachelor's degree in Computer Science...",
                },
            }
        }


# Editing Response Model (inherits from JSONResponse)
class EditingResponseModel(JSONResponse):
    optimized_text: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Text editing processed successfully.",
                "optimized_text": "This is the optimized text after editing.",
            }
        }
