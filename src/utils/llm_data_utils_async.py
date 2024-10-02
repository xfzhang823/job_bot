import os
from dotenv import load_dotenv
import ollama
import openai
from openai import OpenAI, AsyncOpenAI
import asyncio
import logging
import logging_config
import aiohttp
import json
from pydantic import ValidationError
import pandas as pd
from io import StringIO
from typing import Union
from utils.llm_data_utils import get_openai_api_key, clean_and_extract_json
from models.base_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponseModel,
    JobSiteResponseModel,
)


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
    client=None,
    model_id: str = "gpt-4-turbo",
    expected_res_type: str = "str",
    context_type: str = None,  # Use this to determine which JSON model to use
    temperature: float = 0.4,
    max_tokens: int = 1056,
) -> Union[
    TextResponse,
    JSONResponse,
    TabularResponse,
    CodeResponse,
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """
    Handles async API call to OpenAI to generate responses based on a given prompt and expected response type.

    Args:
        - client: OpenAI API client instance (optional)
        - model_id (str): Model ID to use for the OpenAI API call.
        - prompt (str): The prompt to send to the API.
        - expected_res_type (str): The expected type of response from the API
        ('str', 'json', 'tabular', or 'code').
        - context_type (str): Specifies whether to use a job-related JSON model or
        an editing model for JSON ("editing" or "job_site")
        - temperature (str): creativity of the response (0 to 1.0)
        - max_tokens: Maximum tokens to be generated in response.

    Returns:
        Union[str, dict, pd.DataFrame]: Response formatted according to the expected_res_type.
    """
    if not client:
        openai_api_key = get_openai_api_key()
        client = AsyncOpenAI(api_key=openai_api_key)
        logger.info("AsyncOpenAI API instantiated.")

    try:
        logger.info(
            f"Making async API call with expected response type: {expected_res_type}"
        )

        # Make the API request
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who strictly adheres to the provided instructions "
                    "and returns responses in the specified format.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,  # Ensure sufficient tokens for response
        )

        # Extract the content from the response
        response_content = response.choices[0].message.content.strip()
        logger.info(f"Raw LLM Response: {response_content}")

        # Check if the response is empty
        if not response_content:
            logger.error("Received an empty response from OpenAI API.")
            raise ValueError("Received an empty response from OpenAI API.")

        # Handle the response based on expected type
        if expected_res_type == "str":
            parsed_response = TextResponse(content=response_content)
            return parsed_response.content  # Return as plain string

        elif expected_res_type == "json":
            try:
                cleaned_response_content = clean_and_extract_json(response_content)
                if not cleaned_response_content:
                    logger.error("Received an empty response after cleaning.")
                    raise ValueError("Received an empty response after cleaning.")

                response_dict = json.loads(cleaned_response_content)

                # Determine the correct model to use based on context_type
                if context_type == "editing":
                    return EditingResponseModel(
                        optimized_text=response_dict.get("optimized_text")
                    )

                elif context_type == "job_site":
                    return JobSiteResponseModel(
                        job_title=response_dict.get("job_title"),
                        company=response_dict.get("company"),
                        location=response_dict.get("location"),
                        salary_info=response_dict.get("salary_info"),
                        posted_date=response_dict.get("posted_date"),
                        content=response_dict.get("content"),
                    )

                else:
                    # Fallback to a more generic JSON response if no specific context is provided
                    return JSONResponse(data=response_dict)

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse JSON or validate with Pydantic: {e}")
                raise ValueError("Invalid JSON format received from OpenAI API.")

        elif expected_res_type == "tabular":
            try:
                # Parse tabular response using pandas
                df = pd.read_csv(StringIO(response_content))
                parsed_response = TabularResponse(data=df)
                return parsed_response
            except Exception as e:
                logger.error(f"Error parsing tabular data: {e}")
                raise ValueError("Invalid tabular format received from OpenAI API.")

        elif expected_res_type == "code":
            parsed_response = CodeResponse(code=response_content)
            return parsed_response  # return pydantic obj.
        else:
            logger.error(f"Unsupported expected_response_type: {expected_res_type}")
            raise ValueError(f"Unsupported expected_response_type: {expected_res_type}")

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from OpenAI API: {e}")
    except Exception as e:
        logger.error(f"OpenAI API async call failed: {e}")
        raise
