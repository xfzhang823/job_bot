"""
File: base_models.py
Last Updated on:

pydantic base models
"""

from pydantic import BaseModel, ValidationError, constr
from typing import List, Union, Any
import pandas as pd
import json
import logging

# Define Pydantic Models for Different Expected Response Types


class TextResponse(BaseModel):
    content: str


class JSONResponse(BaseModel):
    optimized_text: str  # Example field, modify based on your expected structure

    class Config:
        arbitrary_types_allowed = True  # Allow non-standard types like DataFrame


class TabularResponse(BaseModel):
    data: pd.DataFrame  # Pandas DataFrame for tabular data

    class Config:
        arbitrary_types_allowed = True  # Allow non-standard types like DataFrame


class CodeResponse(BaseModel):
    code: str

    class Config:
        arbitrary_types_allowed = True
