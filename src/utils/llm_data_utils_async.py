import os
from dotenv import load_dotenv
import ollama
import openai
import asyncio
import logging
import aiohttp
import json
from pydantic import ValidationError
import pandas as pd
from io import StringIO
from utils.llm_data_utils import get_openai_api_key
from base_models import CodeResponse, JSONResponse, TabularResponse, TextResponse
import logging_config

logger = logging.getLogger(__name__)


# async def call_llama3_async(
#     prompt: str,
#     expected_res_type: str = "str",
#     temperature: float = 0.4,
#     max_tokens: int = 1056,
# ):
#     try:
#         loop = asyncio.get_running_loop()
#         response = await loop.run_in_executor(
#             None,
#             ollama.generate,
#             model="llama3",
#             prompt=prompt,
#             options={"temperature": temperature, "max_tokens": max_tokens},
#         )
#         # Process response...
#     except asyncio.TimeoutError:
#         logger.error("LLaMA 3 call timed out.")
#         raise
#     except Exception as e:
#         logger.error(f"LLaMA 3 call failed: {e}")
#         raise


async def call_openai_api_async(
    prompt: str,
    model_id="gpt-4-turbo",
    expected_res_type="str",
    temperature=0.4,
    max_tokens=1056,
) -> dict:
    """
    Asynchronous function to handle API call to OpenAI to generate responses based on
    a given prompt and expected response type.

    Args:
        model_id (str): Model ID to use for the OpenAI API call.
        prompt (str): The prompt to send to the API.
        expected_response_type (str): The expected type of response from the API.
                                      Options are 'str' (default), 'json', 'tabular', or 'code'.
        max_tokens: default to 1056

    Returns:
        - Union[str, JSONResponse, pd.DataFrame, CodeResponse]: The response formatted
        according to the specified expected_response_type.
    """
    openai_api_key = get_openai_api_key()  # Get API key as usual
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    # Construct the request payload
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant who strictly adheres to the provided instructions "
                "and returns responses in the specified format.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        logger.info(
            f"Making async API call with expected response type: {expected_res_type}"
        )

        # Async call to OpenAI API
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    logger.error(
                        f"OpenAI API call failed with status code: {response.status}"
                    )
                    raise Exception(
                        f"API call failed with status code {response.status}"
                    )

                response_data = await response.json()
                response_content = response_data["choices"][0]["message"][
                    "content"
                ].strip()

                logger.info(f"Raw LLM Response: {response_content}")

                if not response_content:
                    logger.error("Received an empty response from OpenAI API.")
                    raise ValueError("Received an empty response from OpenAI API.")

                # Handle response based on expected type
                if expected_res_type == "str":
                    parsed_response = TextResponse(content=response_content)
                    return (
                        parsed_response.content
                    )  # return as plain string instead of the model

                elif expected_res_type == "json":
                    try:
                        # Clean and extract JSON if necessary
                        if not response_content.endswith("}"):
                            logger.error("Response appears incomplete or malformed.")
                            raise ValueError(
                                "Received an incomplete response from OpenAI API."
                            )

                        cleaned_response_content = clean_and_extract_json(
                            response_content
                        )

                        if not cleaned_response_content:
                            logger.error("Received an empty response from LLaMA API.")
                            raise ValueError(
                                "Received an empty response from LLaMA API."
                            )

                        response_dict = json.loads(cleaned_response_content)
                        parsed_response = JSONResponse(**response_dict)
                        return parsed_response
                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.error(
                            f"Failed to parse JSON or validate with Pydantic: {e}"
                        )
                        raise ValueError(
                            "Invalid JSON format received from OpenAI API."
                        )

                elif expected_res_type == "tabular":
                    try:
                        # Parse tabular response
                        df = pd.read_csv(StringIO(response_content))
                        parsed_response = TabularResponse(data=df)
                        return parsed_response
                    except Exception as e:
                        logger.error(f"Error parsing tabular data: {e}")
                        raise ValueError(
                            "Invalid tabular format received from OpenAI API."
                        )

                elif expected_res_type == "code":
                    parsed_response = CodeResponse(code=response_content)
                    return parsed_response

                else:
                    logger.error(
                        f"Unsupported expected_response_type: {expected_res_type}"
                    )
                    raise ValueError(
                        f"Unsupported expected_response_type: {expected_res_type}"
                    )

    except Exception as e:
        logger.error(f"OpenAI API async call failed: {e}")
        raise
