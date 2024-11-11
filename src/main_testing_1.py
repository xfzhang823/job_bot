from typing import Dict, Union, Any
from pydantic import BaseModel, ValidationError
import logging
import logging_config
from models.llm_response_models import EditingData, EditingResponseModel, JSONResponse

logger = logging.getLogger(__name__)


# class EditingData(BaseModel):
#     optimized_text: str


# class EditingResponseModel(BaseModel):
#     data: EditingData


# class JSONResponse(BaseModel):
#     data: Dict[str, Any]


def validate_json_type(
    response_model: JSONResponse, json_type: str
) -> Union[EditingResponseModel, JSONResponse]:
    """
    Validates JSON data against a specific Pydantic model based on 'json_type'.
    """
    json_model_mapping = {
        "editing": EditingResponseModel,
        "generic": JSONResponse,
    }

    model = json_model_mapping.get(json_type)
    if not model:
        raise ValueError(f"Unsupported json_type: {json_type}")

    try:
        if json_type == "editing":
            # For single optimized_text response, wrap it in a dictionary with a default key
            if (
                isinstance(response_model.data, dict)
                and "optimized_text" in response_model.data
            ):
                editing_data = {
                    "default": EditingData(
                        optimized_text=response_model.data["optimized_text"]
                    )
                }
                validated_model = EditingResponseModel(data=editing_data)
            # For multiple items or already properly structured data
            else:
                # Convert each value to EditingData instance
                editing_data = {
                    key: EditingData(**value) if isinstance(value, dict) else value
                    for key, value in response_model.data.items()
                }
                validated_model = EditingResponseModel(data=editing_data)
        else:
            validated_model = model(**response_model.model_dump())

        return validated_model

    except ValidationError as e:
        logger.error(f"Validation failed for {json_type}: {e}")
        raise ValueError(f"Invalid format for {json_type}: {e}")


# Test cases
def test_validation():
    # Test Case 1: Single optimized_text
    test_data1 = {
        "data": {"optimized_text": "Developed strategic business analysis reports..."}
    }

    # Test Case 2: Multiple items
    test_data2 = {
        "data": {
            "0.pie_in_the_sky.0": {
                "optimized_text": "This is the optimized text after editing."
            },
            "1.down_to_earth.0": {"optimized_text": "Another piece of optimized text."},
        }
    }

    # Test both cases
    for i, test_data in enumerate([test_data1, test_data2], 1):
        print(f"\nTesting case {i}:")
        try:
            initial_response = JSONResponse(**test_data)
            print(f"Initial response {i}:", initial_response.model_dump())

            validated = validate_json_type(initial_response, "editing")
            print(f"Validated model {i}:", validated.model_dump())
        except Exception as e:
            print(f"Error in case {i}: {e}")


# Run the tests
test_validation()
