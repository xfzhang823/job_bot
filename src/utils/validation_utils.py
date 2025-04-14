from typing import Dict, Any, Mapping
import json
import jsonschema
from jsonschema import validate
from typing import Type, TypeVar, Any
from pydantic import BaseModel, ValidationError
import pandas as pd
import logging
import logging_config
from models.resume_job_description_io_models import SimilarityMetrics


logger = logging.getLogger(__name__)


def validate_json_response(response: str, schema: Mapping) -> Dict[str, Any]:
    """
    Validate a JSON response against a provided schema.

    Args:
        response (str): The JSON response as a string.
        schema (Mapping): The JSON schema to validate against.

    Returns:
        Dict[str, Any]: The validated JSON object.

    Raises:
        ValueError: If the response is not valid JSON or does not conform to the schema.
    """
    try:
        # Load the JSON response
        json_data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e

    try:
        # Validate the JSON data against the schema
        validate(instance=json_data, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"JSON schema validation failed: {e}") from e

    return json_data


def validate_dataframe_with_pydantic(df: pd.DataFrame):
    validated_rows = []
    for index, row in df.iterrows():
        try:
            # Convert the row to a dictionary
            row_dict = row.to_dict()

            # Validate the row using the Pydantic model
            validated_row = SimilarityMetrics(**row_dict)
            validated_rows.append(validated_row.dict())  # Keep validated rows

        except ValidationError as e:
            # Log the error with details about the key and value causing the issue
            logger.error(
                f"Validation failed for row {index} - {row_dict}: {e.errors()}"
            )
            for error in e.errors():
                field = error["loc"][0]  # This is the key (field) causing the error
                msg = error["msg"]  # The validation error message
                logger.error(f"Field '{field}' failed validation: {msg}")
            continue  # Skip invalid rows

    # Return the validated rows as a DataFrame
    return pd.DataFrame(validated_rows)


T = TypeVar("T", bound=BaseModel)


def validate_or_log(
    model_cls: Type[T], data: Any, *, context: str = "unknown"
) -> T | None:
    """
    Attempts to validate `data` against the given Pydantic model class.

    If validation fails, logs a warning with the context and returns None.

    Args:
        model_cls (Type[T]): The Pydantic model class to validate against.
        data (Any): The input data (typically a dict).
        context (str): A string for logging context (e.g. URL or filename).

    Returns:
        T | None: The validated Pydantic model, or None if validation failed.
    """
    try:
        return model_cls.model_validate(data)  # type: ignore[attr-defined]
    except ValidationError as e:
        logger.warning(
            f"⚠️ Validation failed for {model_cls.__name__} in context: {context}\n{e}"
        )
        return None
