"""
Module name: resume_job_description_io_models.py
Last updated on: 2024-10-16
"""

# Import dependencies
from pathlib import Path
from typing import Dict, Optional, Union
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_serializer,
    field_validator,
    DirectoryPath,
    RootModel,
)
from pydantic_core import Url
import logging
import logging_config


# Set up logger
logger = logging.getLogger(__name__)


# Model to validate responsibility input
class Responsibilities(BaseModel):
    """
    Pydantic model for validating flattened responsibilities.
    Ensures all responsibility JSON files contain both a `url` (unique identifier)
    and `responsibilities` (flattened responsibilities).

    Attributes:
        - url (Optional[str]): The job posting URL (optional).
        Defaults to "Not Available" if missing.
        - responsibilities (Dict[str, str]): A dictionary containing flattened
        responsibilities.

    Example Usage:
        ```python
        responsibilities_input = {
            "url": "https://example.com/job/software-engineer",
            "responsibilities": {
                "0.responsibilities.0": "Developed AI-driven marketing strategies.",
                "0.responsibilities.1": "Led a team of 10 data scientists."
            }
        }

        validated_responsibilities = Responsibilities(**responsibilities_input)

        print(validated_responsibilities.url)  # Output: "https://example.com/job/software-engineer"
        print(validated_responsibilities.responsibilities["0.responsibilities.0"])
        # Output: "Developed AI-driven marketing strategies."
        ```

    Example JSON Output:
        ```json
        {
            "url": "https://example.com/job/software-engineer",
            "responsibilities": {
                "0.responsibilities.0": "Developed AI-driven marketing strategies.",
                "0.responsibilities.1": "Led a team of 10 data scientists."
            }
        }
        ```

    Raises:
        - `ValidationError`: If `responsibilities` is missing or not a dictionary.
    """

    url: Optional[str] = Field(None, description="URL of the job posting (optional).")
    responsibilities: Dict[str, str] = Field(
        ..., description="Flattened job responsibilities."
    )


# Model to validate the initial output: optimized text for a single responsibility
# to requirement match
class OptimizedText(BaseModel):
    """
    Pydantic base model for higher level models such as ResponsibilityMatch.

    Exmaple data:
        {
            "optimized_text": "Led strategic initiatives for a leading multinational IT corporation."
        }
    """

    optimized_text: str


# Model to validate the next output: optimized texts for multiple requirements
class ResponsibilityMatch(BaseModel):
    """
    Pydantic model that validates the structure mapping requirement_keys to
    their corresponding optimized text objects.

    Example Output:
        A dictionary where:
        - Keys are `requirement_key`s (e.g., "0.pie_in_the_sky.0").
        - Values are objects containing the optimized text for the corresponding requirement.

    Structure:
        {
            "0.pie_in_the_sky.0": {
                "optimized_text": "Led strategic initiatives for a leading multinational IT corporation."
            },
            "0.pie_in_the_sky.1": {
                "optimized_text": "Optimized the service partner ecosystem for a major global IT vendor."
            }
        }

    Attributes:
        - optimized_by_requirements (Dict[str, OptimizedText]):
            A dictionary mapping requirement keys to their corresponding `OptimizedText` objects.
    """

    optimized_by_requirements: Dict[str, OptimizedText]


# *Model to validate final output: multiple resp to requirements matches
class ResponsibilityMatches(BaseModel):
    """
    Pydantic model for validating nested responsibilities and their associated requirement matches.

    This model is designed to handle the hierarchical structure of responsibilities and
    their corresponding requirement-based optimized texts. It validates the format of
    nested JSON files where responsibilities map to multiple requirement-based optimizations.

    Main purpose: validate I/O in the resume editing process, and validate responsibilities
    json files in iteration 1 and beyond.

    Structure Overview:
    - At the top level, the keys represent `responsibility_key`s (e.g., "0.responsibilities.0").
    - Each responsibility key maps to a dictionary of `requirement_key`s.
    - Each `requirement_key` maps to an object containing the `optimized_text`.

    Example JSON Structure:
    ```json
    {
        "0.responsibilities.0": {
            "0.pie_in_the_sky.0": {
                "optimized_text": "Led strategic initiatives for a leading multinational IT
                corporation..."
            },
            "0.pie_in_the_sky.1": {
                "optimized_text": "Optimized the service partner ecosystem for
                a major global IT vendor..."
            }
        },
        "0.responsibilities.1": {
            "1.down_to_earth.0": {
                "optimized_text": "11 years of experience providing strategic insights
                to a major global IT vendor..."
            },
            "1.down_to_earth.1": {
                "optimized_text": "Experience optimizing the service partner ecosystem
                in Asia Pacific..."
            }
        }
    }
    ```

    Attributes:
    - responsibilities (Dict[str, ResponsibilityMatch]):
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
        Maps `requirement_key`s (e.g., "0.pie_in_the_sky.0") to instances of
        `OptimizedText`.

    This model ensures the correct format for nested responsibilities and their
    corresponding requirement matches, facilitating easier processing and validation
    of complex nested data.
    """

    responsibilities: Dict[str, ResponsibilityMatch] = Field(
        ..., description="Mapping from responsibilty keys to nested requirement matches"
    )


class NestedResponsibilities(BaseModel):
    """
    Pydantic model for validating job responsibilities after iteration 1.

    This model is designed to support the structured representation of responsibilities
    after they have been aligned with job requirements. It removes unnecessary nesting,
    making it easier to validate and process responsibilities efficiently.

    Structure Overview:
    - `url`: Stores the job posting URL (string or `HttpUrl`).
    - `responsibilities`: A dictionary mapping responsibility keys
    (e.g., `"0.responsibilities.0"`) to `ResponsibilityMatch` objects, which contain
    optimized texts for various job requirements.

    Example JSON Structure:
    ```json
    {
        "url": "https://careers.example.com/job/12345/",
        "responsibilities": {
            "0.responsibilities.0": {
                "optimized_by_requirements": {
                    "1.down_to_earth.0": {
                        "optimized_text": "Led strategic initiatives for a multinational IT corporation."
                    },
                    "1.down_to_earth.1": {
                        "optimized_text": "Developed scalable solutions for optimizing service partnerships."
                    }
                }
            },
            "0.responsibilities.1": {
                "optimized_by_requirements": {
                    "2.other.0": {
                        "optimized_text": "Managed cross-functional projects for international expansion."
                    }
                }
            }
        }
    }
    ```

    Attributes:
    - `url` (`str` | `HttpUrl`): The URL of the job posting, validated as a proper URL.
    - `responsibilities` (`Dict[str, ResponsibilityMatch]`): A dictionary where:
        - Keys are responsibility identifiers (e.g., `"0.responsibilities.0"`).
        - Values are `ResponsibilityMatch` objects containing requirement-based
        optimizations.

    Usage Example:
    ```python
    from pydantic import ValidationError
    import json

    # Load the JSON data
    with open("responsibilities.json", "r") as f:
        json_data = json.load(f)

    try:
        # Validate the JSON structure using the NestedResponsibilities model
        validated_data = NestedResponsibilities.model_validate(json_data)
        print("Validation successful:", validated_data)

    except ValidationError as e:
        print(f"Validation error: {e}")
    ```

    Notes:
    - This model replaces the previous `ResponsibilityMatches` wrapper with a
    **direct dictionary mapping**.
    - The `coerce_url_to_str` method ensures that `HttpUrl` objects are converted
    to strings before validation.
    - The model is designed for **iteration 1 and beyond**, where responsibilities
    have been refined based on job requirements.
    """

    url: str | HttpUrl = Field(
        ...,
        description="URL of the job posting (required). Can be a string or an HttpUrl object.",
    )
    responsibilities: Dict[
        str, ResponsibilityMatch
    ]  # ✅ Simulates ResponsibilityMatches (direct ResponsibilityMatches creates
    # an unwanted "wrapper")

    @field_validator("url", mode="before")  # Ensures URLs are always strings
    def coerce_url_to_str(cls, v):  # pylint: disable=no-self-argument
        """
        Convert `HttpUrl` or `Url` objects to a string before validation.

        This ensures that any URL stored in the `url` field is always treated as a string,
        avoiding issues with serialization and downstream processing.
        """
        if isinstance(v, Url):
            return str(v)
        return v  # Leave strings or other types as they are

    @field_serializer("url")
    def serialize_url(self, v: str | HttpUrl) -> str:
        """
        Serializer ensures that when calling model_dump() or model.json(),
        the URL field is converted to a string.
        """
        return str(v)


# Model to validate requirement input
class Requirements(BaseModel):
    """
    Pydantic model for validating job requirements.
    Ensures all job requirement files contain both a `url` (unique identifier)
    and `requirements` (flattened job qualifications).

    Attributes:
        url (str): The job posting URL (required).
        requirements (Dict[str, str]): A dictionary containing flattened job requirements.

    Example Usage:
        ```python
        # Example input JSON structure
        requirements_input = {
            "url": "https://example.com/job/software-engineer",
            "requirements": {
                "0.pie_in_the_sky.0": "10+ years of experience in B2B SaaS leadership.",
                "1.down_to_earth.0": "3+ years experience in software development.",
                "2.bare_minimum.0": "Bachelor's degree in Computer Science or related field."
            }
        }

        # Load the input data into the Pydantic model for validation
        validated_requirements = Requirements(**requirements_input)

        # Accessing validated data
        print(validated_requirements.url)  # Output: "https://example.com/job/software-engineer"
        print(validated_requirements.requirements["0.pie_in_the_sky.0"])
        # Output: "10+ years of experience in B2B SaaS leadership."
        ```

    Example JSON Output:
        ```json
        {
            "url": "https://example.com/job/software-engineer",
            "requirements": {
                "0.pie_in_the_sky.0": "10+ years of experience in B2B SaaS leadership.",
                "1.down_to_earth.0": "3+ years experience in software development.",
                "2.bare_minimum.0": "Bachelor's degree in Computer Science or related field."
            }
        }
        ```

    Raises:
        - `ValidationError`: If `url` is missing or not a string.
        - `ValidationError`: If `requirements` is missing or not a dictionary.
    """

    url: str = Field(..., description="URL of the job posting (required).")
    requirements: Dict[str, str] = Field(..., description="Flattened job requirements.")


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

    job_posting_url: str = Field(..., description="Job posting url")
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

    @field_validator("*", mode="before", check_fields=False)
    def clean_strings(cls, v):  # ignore the error
        """
        Clean string fields by removing newline characters and stripping whitespace.
        """
        if isinstance(v, str):
            return v.replace("\n", " ").strip()
        return v


# Inner model for the file paths associated with each job posting URL
class JobFilePaths(BaseModel):
    """Inner model as part of the JobFileMappings class"""

    reqs: Union[str, Path] = Field(
        ..., description="Path to the flattened requirements JSON file"
    )
    resps: Union[str, Path] = Field(
        ..., description="Path to the flattened responsibilities JSON file"
    )
    sim_metrics: Union[str, Path] = Field(
        ..., description="Path to the similarity metrics CSV file"
    )
    pruned_resps: Union[str, Path] = Field(
        ..., description="Path to the pruned responsibilities JSON file"
    )


# *Outer model for the mapping of job URLs to JobFilePaths
class JobFileMappings(RootModel[Dict[HttpUrl, JobFilePaths]]):
    """
    Pydantic model for validating a mapping of job URLs to associated file paths for
    requirements, responsibilities, and metrics.

    This model maps job URLs directly to their corresponding file paths without relying on
    the `__root__` attribute, using a dictionary structure where job posting URLs are keys,
    and the associated file paths are values validated by the `JobFilePaths` model.

    Structure Overview:
    - The top-level keys represent job URLs (validated as `HttpUrl`).
    - The values for each URL are instances of the `JobFilePaths` model containing:
      - 'reqs': The JSON file containing flattened job requirements.
      - 'resps': The JSON file with (flattened or nested) job responsibilities.
      - 'sim_metrics': The CSV file containing similarity metrics between responsibilities
        and requirements.
      - 'pruned_resps': The JSON file containing pruned job responsibilities.

    Example JSON Structure:
    ```json
    {
        "https://www.example.com/job/123": {
            "reqs": "C:\\path\\to\\requirements.json",
            "resps": "C:\\path\\to\\responsibilities.json",
            "sim_metrics": "C:\\path\\to\\similarity_metrics.csv",
            "pruned_resps": "C:\\path\\to\\pruned_responsibilities.json"
        },
        "https://www.anotherexample.com/job/456": {
            "reqs": "C:\\path\\to\\another_requirements.json",
            "resps": "C:\\path\\to\\another_responsibilities.json",
            "sim_metrics": "C:\\path\\to\\another_similarity_metrics.csv",
            "pruned_resps": "C:\\path\\to\\another_pruned_responsibilities.json"
        }
    }
    ```

    Attributes:
    - A dictionary mapping job posting URLs (validated as `HttpUrl`) to file paths.
      Each file path dictionary is represented by the `JobFilePaths` model,
      which validates the paths to requirements, responsibilities, similarity metrics,
      and pruned responsibilities.

    Example Usage:
    ```python
    from pydantic import ValidationError, HttpUrl
    from pathlib import Path

    # Example data
    example_data = {
        "https://www.example.com/job/123": {
            "reqs": "C:\\path\\to\\requirements.json",
            "resps": "C:\\path\\to\\responsibilities.json",
            "sim_metrics": "C:\\path\\to\\similarity_metrics.csv",
            "pruned_resps": "C:\\path\\to\\pruned_responsibilities.json"
        },
        "https://www.anotherexample.com/job/456": {
            "reqs": "C:\\path\\to\\another_requirements.json",
            "resps": "C:\\path\\to\\another_responsibilities.json",
            "sim_metrics": "C:\\path\\to\\another_similarity_metrics.csv",
            "pruned_resps": "C:\\path\\to\\another_pruned_responsibilities.json"
        }
    }

    try:
        # Validate the data using the JobFileMappings model
        validated_data = JobFileMappings.parse_obj(example_data)
        print("Validation successful:", validated_data)
    except ValidationError as e:
        print(f"Validation error: {e}")
    ```

    Models:
    - **JobFilePaths**:
        This model validates the file paths for requirements, responsibilities,
        similarity metrics, and pruned responsibilities.
        It supports both `str` and `Path` types for flexibility.
    - **HttpUrl**:
        The job posting URLs are validated as `HttpUrl` to ensure they are properly
        formatted and valid.

    This model provides a structured way to validate mappings of job URLs to their
    associated file paths, ensuring the data conforms to the expected structure and types.
    """

    pass  # RootModel does not require additional fields


class PipelineInput(BaseModel):
    """Validate file and directory paths - if exist(s)"""

    data_directory: DirectoryPath
