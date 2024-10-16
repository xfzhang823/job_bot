"""
Module name: resume_job_description_io_models.py
Last updated on: 2024-10-16
"""

from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, HttpUrl


# Model to validate the initial output: optimized text for a single responsibility
# to requirement match
class Requirement(BaseModel):
    req_key: str = Field(..., description="Unique key for the requirement")
    req_text: str = Field(..., description="Text of the requirement")


class Responsibility(BaseModel):
    resp_key: str = Field(..., description="Unique key for the responsibility")
    resp_text: str = Field(..., description="Text of the responsibility")


class OptimizedText(BaseModel):
    optimized_text: str


# Model to validate the next output: optimized texts for multiple requirements
class ResponsibilityMatch(BaseModel):
    optimized_by_requirements: Dict[str, OptimizedText]


# *Model to validate final output: multiple resp to requirements matches
class ResponsibilityMatches(BaseModel):
    """
    Pydantic model for validating nested responsibilities and their associated requirement matches.

    This model is designed to handle the hierarchical structure of responsibilities and
    their corresponding requirement-based optimized texts. It validates the format of
    nested JSON files where responsibilities map to multiple requirement-based optimizations.

    Main purpose: validate I/O in the resume editing process, and validate responsibilities json files 
    in iteration 1 and beyond.

    Structure Overview:
    - At the top level, the keys represent `responsibility_key`s (e.g., "0.responsibilities.0").
    - Each responsibility key maps to a dictionary of `requirement_key`s.
    - Each `requirement_key` maps to an object containing the `optimized_text`.

    Example JSON Structure:
    ```json
    {
        "0.responsibilities.0": {
            "0.pie_in_the_sky.0": {
                "optimized_text": "Led strategic initiatives for a leading multinational IT corporation..."
            },
            "0.pie_in_the_sky.1": {
                "optimized_text": "Optimized the service partner ecosystem for a major global IT vendor..."
            }
        },
        "0.responsibilities.1": {
            "1.down_to_earth.0": {
                "optimized_text": "11 years of experience providing strategic insights to a major \
                    global IT vendor..."
            },
            "1.down_to_earth.1": {
                "optimized_text": "Experience optimizing the service partner ecosystem in \
                    Asia Pacific..."
            }
        }
    }
    ```

    Attributes:
    - **responsibilities (Dict[str, ResponsibilityMatch])**:
        A dictionary where:
        - The keys represent `responsibility_key`s (e.g., "0.responsibilities.0").
        - The values are instances of `ResponsibilityMatch` that contain the
          nested dictionary of `requirement_key`s to `OptimizedText` mappings.

    Usage Example:
    ```python
    from pydantic import ValidationError
    import json

    # Load the JSON data from a file
    with open("path_to_responsibilities_json_file.json", 'r') as f:
        json_data = json.load(f)

    try:
        # Validate the JSON structure using the ResponsibilityMatches model
        validated_data = ResponsibilityMatches.parse_obj({"responsibilities": json_data})
        print("Validation successful:", validated_data)

    except ValidationError as e:
        print(f"Validation error: {e}")
    ```

    Models Used:
    - OptimizedText:
        Validates the inner `optimized_text` for each requirement.
        Contains a single string field for the text itself.
    - ResponsibilityMatch:
        Maps `requirement_key`s (e.g., "0.pie_in_the_sky.0") to instances of `OptimizedText`.

    This model ensures the correct format for nested responsibilities and their corresponding
    requirement matches, facilitating easier processing and validation of complex nested data.
    """

    responsibilities: Dict[str, ResponsibilityMatch] = Field(
        ..., description="Mapping from responsibilty keys to nested requirement matches"
    )


# Model to validate responsibility input
class ResponsibilityInput(BaseModel):
    responsibilities: Dict[str, str]


# Model to validate requirement input
class RequirementsInput(BaseModel):
    requirements: Dict[str, str]


# *Model validate Similarity Metrics
class SimilarityMetrics(BaseModel):
    """
    Pydantic model for validating similarity metrics fields in CSV files.

    This model is used to validate data rows from a CSV file that contains
    various similarity metrics between job responsibilities and job requirements.
    It ensures that required fields are present and properly typed, while
    optional fields can be included if necessary.

    Key Features:
    - Enforces type validation for both required and optional fields.
    - Ensures that each row in the CSV file conforms to the expected structure
    (e.g., string for text fields, float for metric fields).
    - Optional fields can be left as `None` if not present in the data.

    Note:
    There is no need to specify file types explicitly when loading CSV files with Pandas,
    as this model operates on the loaded data. It is assumed that the data is provided as
    Python dictionaries, such as the rows returned by `pandas.DataFrame.iterrows()`.

    Usage Example:

    ```python
    import pandas as pd
    from pydantic import ValidationError

    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv("path_to_similarity_metrics.csv")

    # Validate each row
    for index, row in df.iterrows():
        try:
            # Instantiate the SimilarityMetrics model for each row
            similarity_metrics = SimilarityMetrics(
                responsibility_key=row['responsibility_key'],
                responsibility=row['responsibility'],
                requirement_key=row['requirement_key'],
                requirement=row['requirement'],
                bert_score_precision=row['bert_score_precision'],
                soft_similarity=row['soft_similarity'],
                word_movers_distance=row['word_movers_distance'],
                deberta_entailment_score=row['deberta_entailment_score'],

                # Optional fields (if present in the row)
                bert_score_precision_cat=row.get('bert_score_precision_cat'),
                soft_similarity_cat=row.get('soft_similarity_cat'),
                word_movers_distance_cat=row.get('word_movers_distance_cat'),
                deberta_entailment_score_cat=row.get('deberta_entailment_score_cat'),
                scaled_bert_score_precision=row.get('scaled_bert_score_precision'),
                scaled_deberta_entailment_score=row.get('scaled_deberta_entailment_score'),
                scaled_soft_similarity=row.get('scaled_soft_similarity'),
                scaled_word_movers_distance=row.get('scaled_word_movers_distance'),
                composite_score=row.get('composite_score'),
                pca_score=row.get('pca_score')
            )
            print(f"Row {index} passed validation:", similarity_metrics)

        except ValidationError as e:
            print(f"Validation error in row {index}: {e}")

    # Example Output:
    # Validation error in row 5: 1 validation error for SimilarityMetrics
    # bert_score_precision
    #    value is not a valid float (type=type_error.float)
    """

    responsibility_key: str = Field(..., description="Responsibility key")
    responsibility: str = Field(..., description="Responsibility text")
    requirement_key: str = Field(..., description="Requirement key")
    requirement: str = Field(..., description="Requirement text")
    bert_score_precision: float = Field(..., description="BERTScore precision")
    soft_similarity: float = Field(..., description="Soft similarity")
    word_movers_distance: float = Field(..., description="Word Mover's Distance")
    deberta_entailment_score: float = Field(..., description="Deberta entailment score")

    # Optional fields
    bert_score_precision_cat: Optional[str] = Field(
        None, description="BERTScore precision category"
    )
    soft_similarity_cat: Optional[str] = Field(
        None, description="Soft similarity category"
    )
    word_movers_distance_cat: Optional[str] = Field(
        None, description="Word Mover's Distance category"
    )
    deberta_entailment_score_cat: Optional[str] = Field(
        None, description="Deberta entailment score category"
    )
    scaled_bert_score_precision: Optional[float] = Field(
        None, description="Scaled BERT score"
    )
    scaled_deberta_entailment_score: Optional[float] = Field(
        None, description="Scaled Deberta entailment score"
    )
    scaled_soft_similarity: Optional[float] = Field(
        None, description="Scaled soft similarity"
    )
    scaled_word_movers_distance: Optional[float] = Field(
        None, description="Scaled Word Mover's Distance"
    )
    composite_score: Optional[float] = Field(None, description="Composite score")
    pca_score: Optional[float] = Field(None, description="PCA score")


# Inner model for the file paths associated with each job posting URL
class JobFilePaths(BaseModel):
    reqs: Path = Field(..., description="Path to the flattened requirements JSON file")
    resps: Path = Field(
        ..., description="Path to the flattened responsibilities JSON file"
    )
    sim_metrics: Path = Field(
        ..., description="Path to the similarity metrics CSV file"
    )
    pruned_resps: Path = Field(
        ..., description="Path to the pruned responsibilities JSON file"
    )


# *Outer model for the mapping of job URLs to JobFilePaths
class JobFileMappings(BaseModel):
    """
    Pydantic model for validating a mapping of job URLs to associated file paths for
    requirements, responsibilities, and metrics.

    This model is used to validate the structure of a nested JSON format
    where job URLs map to file paths for related data, such as
    requirements, responsibilities, similarity metrics, and pruned responsibilities.

    Structure Overview:
    - The top-level keys represent job URLs (e.g., URLs to job postings).
    - The values for each URL are dictionaries containing paths to:
      - 'reqs': The JSON file containing flattened job requirements.
      - 'resps': The JSON file with (flattened or nested) job responsibilities.
      - 'sim_metrics': The CSV file containing similarity metrics between responsibilities
      and requirements.
      - 'pruned_resps': The JSON file containing pruned job responsibilities.

    Example JSON Structure:
    ```json
    {
        "https://www.example.com/job/123": {
            "reqs_flat": "C:\\path\\to\\requirements.json",
            "resps_flat": "C:\\path\\to\\responsibilities.json",
            "sim_metrics": "C:\\path\\to\\similarity_metrics.csv",
            "pruned_resps_flat": "C:\\path\\to\\pruned_responsibilities.json"
        },
        "https://www.anotherexample.com/job/456": {
            "reqs_flat": "C:\\path\\to\\another_requirements.json",
            "resps_flat": "C:\\path\\to\\another_responsibilities.json",
            "sim_metrics": "C:\\path\\to\\another_similarity_metrics.csv",
            "pruned_resps_flat": "C:\\path\\to\\another_pruned_responsibilities.json"
        }
    }
    ```

    Attributes:
    - files (Dict[HttpUrl, JobFilePaths]):
      A dictionary mapping job posting URLs (validated as `HttpUrl`) to file paths.
      The file paths are represented by the `JobFilePaths` model, which validates the paths
      to requirements, responsibilities, similarity metrics, and pruned responsibilities.

    Example Usage:
    ```python
    from pydantic import ValidationError, HttpUrl
    import json
    from pathlib import Path

    # Example JSON structure (mocked as a Python dictionary here)
    example_data = {
        "https://www.example.com/job/123": {
            "reqs_flat": "C:\\path\\to\\requirements.json",
            "resps_flat": "C:\\path\\to\\responsibilities.json",
            "sim_metrics": "C:\\path\\to\\similarity_metrics.csv",
            "pruned_resps_flat": "C:\\path\\to\\pruned_responsibilities.json"
        },
        "https://www.anotherexample.com/job/456": {
            "reqs_flat": "C:\\path\\to\\another_requirements.json",
            "resps_flat": "C:\\path\\to\\another_responsibilities.json",
            "sim_metrics": "C:\\path\\to\\another_similarity_metrics.csv",
            "pruned_resps_flat": "C:\\path\\to\\another_pruned_responsibilities.json"
        }
    }

    try:
        # Validate the data using the JobMappings model
        validated_data = JobMappings(jobs=example_data)
        print("Validation successful:", validated_data)

    except ValidationError as e:
        print(f"Validation error: {e}")
    ```

    Models:
    - JobFilePaths:
        This model validates the file paths for requirements, responsibilities, similarity metrics,
        and pruned responsibilities.
        Each field is a `Path` object, ensuring that the values provided are valid file paths.
    - HttpUrl:
        The job posting URLs are validated as `HttpUrl` to ensure they are valid and properly formatted.

    This model provides a convenient way to validate mappings of job URLs to their associated file paths,
    ensuring the data conforms to the expected structure and types.
    """

    files: Dict[HttpUrl, JobFilePaths] = Field(
        ..., description="Mapping from job posting URLs to their associated file paths"
    )


# # Example usage:
# responsibilities_input = {
#     "0.responsibilities.0": "Provided strategic insights to a major global IT vendor...",
#     "0.responsibilities.1": "Assisted a U.S.-based international services provider...",
#     # more responsibilities...
# }

# requirements_input = {
#     "0.pie_in_the_sky.0": "10+ years of experience in B2B SaaS...",
#     "0.pie_in_the_sky.1": "Ph.D. in Data Science...",
#     # more requirements...
# }

# # Load the input data into Pydantic models for validation
# validated_responsibilities = ResponsibilityInput(
#     responsibilities=responsibilities_input
# )
# validated_requirements = RequirementsInput(requirements=requirements_input)

# # Accessing validated data
# print(validated_responsibilities.responsibilities["0.responsibilities.0"])
