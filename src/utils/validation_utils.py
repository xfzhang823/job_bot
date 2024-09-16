from typing import Dict, Any, Mapping
import json
import jsonschema
from jsonschema import validate


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
