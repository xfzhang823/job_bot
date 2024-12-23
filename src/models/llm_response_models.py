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


class BaseResponseModel(BaseModel):
    """
    Base model that provides common fields for various response models.

    Attributes:
        status (str): Indicates the success status of the response, defaults to "success".
        message (Optional[str]): Optional field to provide additional feedback or a message.

    Config:
        arbitrary_types_allowed (bool): Allows non-standard types like pandas DataFrame.

    *Allows validation functions to add status and message easily!
    """

    status: str = "success"
    message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class JSONResponse(BaseModel):
    """
    General-purpose model for handling JSON-based responses.

    Attributes:
        data (Union[Dict[str, Any], List[Dict[str, Any]]]): Holds JSON data,
        which can be either a dictionary or a list of dictionaries.

    Config:
        arbitrary_types_allowed (bool): Allows non-standard types in
        JSON responses.
    """

    data: Union[Dict[str, Any], List[Dict[str, Any]]]

    class Config:
        arbitrary_types_allowed = True


class TextResponse(BaseResponseModel):
    """
    Model for plain text responses.

    Attributes:
        content (str): Holds plain text content of the response.

    Config:
        json_schema_extra (dict): Provides an example structure for documentation.
    """

    content: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Text response processed.",
                "content": "This is the plain text content.",
            }
        }


class TabularResponse(BaseResponseModel):
    """
    Model for handling tabular data responses using pandas DataFrame.

    Attributes:
        data (pd.DataFrame): Contains the tabular data.

    Config:
        - arbitrary_types_allowed (bool): Allows DataFrame as a valid type.
        - json_schema_extra (dict): Example structure for tabular data documentation.
    """

    data: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Tabular data processed.",
                "data": "Pandas DataFrame object",
            }
        }


class CodeResponse(BaseResponseModel):
    """
    Model for responses containing code snippets.

    Attributes:
        code (str): Holds the code as a string.

    Config:
        json_schema_extra (dict): Example structure for code response documentation.
    """

    code: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Code response processed.",
                "code": "print('Hello, world!')",
            }
        }


class OptimizedTextData(BaseModel):
    """Inner model to specify required 'optimized_text' field."""

    optimized_text: str = Field(..., description="The optimized text after editing.")


class EditingResponseModel(BaseResponseModel):
    """
    Model for responses involving text editing operations.

    Attributes:
        data (OptimizedTextData): Contains 'optimized_text' as a required field.
    """

    data: OptimizedTextData  # Use inner model to enforce 'optimized_text' structure

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Text editing processed successfully.",
                "data": {"optimized_text": "This is the optimized text after editing."},
            }
        }


class JobSiteData(BaseModel):
    """
    Inner model containing detailed job site information.

    Attributes:
        url (str): The URL of the job posting (required).
        job_title (str): Title of the job position (required).
        company (str): Name of the company posting the job (required).
        location (Optional[str]): Job location.
        salary_info (Optional[str]): Salary information, if available.
        posted_date (Optional[str]): Date when the job was posted.
        content (Optional[Dict[str, Any]]): Contains the job description, responsibilities, and qualifications as a dictionary.
    """

    url: str = Field(..., description="Job posting URL")
    job_title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    salary_info: Optional[str] = Field(None, description="Salary information")
    posted_date: Optional[str] = Field(None, description="Job posting date")
    content: Optional[Dict[str, Any]] = Field(
        None,
        description="Dictionary containing job description, responsibilities, and qualifications",
    )


class JobSiteResponseModel(BaseResponseModel):
    """
    Model for handling job site response data, standardizing job-related information.

    Attributes:
        data (JobSiteData): Holds detailed job site information as a nested JobSiteData instance.

    Config:
        json_schema_extra (dict): Provides an example structure for documentation.
    """

    data: JobSiteData

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Job site data processed successfully.",
                "data": {
                    "url": "https://example.com/job-posting",
                    "job_title": "Software Engineer",
                    "company": "Tech Corp",
                    "location": "San Francisco, CA",
                    "salary_info": "$100,000 - $120,000",
                    "posted_date": "2024-11-08",
                    "content": {
                        "description": "We are looking for a Software Engineer...",
                        "responsibilities": [
                            "Develop software",
                            "Collaborate with team",
                        ],
                        "qualifications": [
                            "BS in Computer Science",
                            "2+ years experience",
                        ],
                    },
                },
            }
        }
