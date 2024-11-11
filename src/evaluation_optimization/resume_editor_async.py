"""
File: resume_editing.py
Author: Xiao-Fei Zhang
Last updated: 2024 Sep 12
"""

import logging
from dotenv import load_dotenv
import os
import json
import jsonschema
import uuid
import logging_config
from pydantic import BaseModel, ValidationError
from typing import Dict, Union, Optional, cast

import openai
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic

from prompts.prompt_templates import (
    SEMANTIC_ALIGNMENT_PROMPT,
    ENTAILMENT_ALIGNMENT_PROMPT,
    SEMANTIC_ENTAILMENT_ALIGNMENT_PROMPT,
    STRUCTURE_TRANSFER_PROMPT,
)
from utils.llm_api_utils import get_openai_api_key
from utils.llm_api_utils_async import (
    call_openai_api_async,
    call_llama3_async,
    call_claude_api_async,
)
from utils.validation_utils import validate_json_response
from models.llm_response_models import (
    EditingResponseModel,
    JobSiteResponseModel,
    TextResponse,
    JSONResponse,
    TabularResponse,
    CodeResponse,
)

# logging
logger = logging.getLogger(__name__)

# Get openai api key
openai.api_key = get_openai_api_key()

# Define the JSON schema somewhere accessible
LLM_RES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "optimized_text": {"type": "string"},
    },
    "required": ["optimized_text"],
}


class TextEditorAsync:
    """
    * Async version of class TextEditor
    A class to edit and optimize a text, given another text to "match" to, using language models
    like OpenAI GPT-4, Claude, or LLaMA3.

    This class provides methods to perform text editing based on semantic entailment
    and dependency parsing alignment using various language models via API calls.

    Attributes:
        llm_provider (str): The model to use for the API calls ('openai', 'claude', or 'llama3').
        client (any): instantiated API for OpenAI API calls.
        model_id (str): The model ID to use for the API calls (e.g., 'gpt-4-turbo').
        temperature (float): Temperature parameter for the model, affecting response variability.
        max_tokens (int): The maximum number of tokens to generate for the model output.

    Methods:
        - generate_text_id(text_id=None): Generates a unique text ID using UUID.
        - format_prompt(prompt_template, content_1, content_2): Formats the prompt using a provided template.
        - call_llm_async(prompt, llm_provider=None, temperature=None):
            Calls the specified LLM API (OpenAI, Anthropic, or LLaMA3) with the given prompt and returns
            the response.
        - validate_response(response_dict): Validates a response dictionary against a predefined JSON schema.
        - edit_for_dp(target_text, source_text, text_id=None, llm_provider=None, temperature=None):
            Edits text to align with source text's dependency parsing (DP).
        - edit_for_entailment(target_text, source_text, text_id=None, llm_provider=None, temperature=None):
            Edits text based on semantic entailment.
        - edit_for_semantics(target_text, source_text, text_id=None, llm_provider=None, temperature=None):
            Edits text to align with source text's semantics.
        - edit_for_semantics_and_entailment(candidate_text, reference_text, text_id=None, llm_provider=None,
          temperature=None): Edits text to align with source text's semantics and strengthen entailment relationships.
    """

    def __init__(
        self,
        llm_provider="openai",
        model_id="gpt-4-turbo",
        temperature=0.7,
        max_tokens=1056,
        client=None,
    ):
        self.llm_provider = llm_provider  # Default model ('openai')
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client  # Store the passed client

        # Conditionally initialize the client based on model
        if self.llm_provider == "openai" and not self.client:
            # Initialize OpenAI API client if it's not provided
            api_key = get_openai_api_key()  # Fetch the API key
            self.client = OpenAI(api_key=api_key)  # Instantiate OpenAI API client
        elif self.llm_provider == "llama3":
            # If using LLaMA3, no OpenAI client initialization is needed
            # You may initialize LLaMA3-specific settings here if needed
            logger.info("Using LLaMA3 model for text editing.")
        elif not self.client:
            raise ValueError(f"Unsupported model: {self.llm_provider}")

    def generate_text_id(self, text_id=None):
        """Generate a unique text_id using UUID if not provided."""
        return text_id or str(uuid.uuid4())

    def format_prompt(self, prompt_template, content_1, content_2):
        """Format the prompt using the provided template."""
        try:
            return prompt_template.format(content_1=content_1, content_2=content_2)
        except KeyError as e:
            logger.error(f"Error formatting prompt: {e}")
            raise

    async def call_llm_async(
        self,
        prompt: str,
        llm_provider: str = "openai",
        temperature: Optional[float] = None,
    ) -> Union[
        JSONResponse,
        TabularResponse,
        TextResponse,
        CodeResponse,
        EditingResponseModel,
        JobSiteResponseModel,
    ]:
        """
        *Async version of the method.

        Call the specified LLM API (OpenAI, Anthropic, or LLaMA3) with the provided prompt
        and return the response.

        Args:
            prompt (str): The formatted prompt to send to the LLM API.
            llm_provider (str): The llm provider to use for the API call ('openai', 'claude', or 'llama3').
            temperature (float, optional): Temperature setting for this specific API call.
                                           If None, uses the class-level temperature.

        Returns:
            Union[JSONResponse, TabularResponse, TextResponse, CodeResponse, EditingResponseModel, JobSiteResponseModel]:
            The structured response from the API, validated if it passes JSON schema requirements.
        """
        temperature = temperature if temperature is not None else self.temperature
        # Choose the appropriate LLM API
        if llm_provider == "openai":
            response_pyd_obj = await call_openai_api_async(
                prompt=prompt,
                model_id=self.model_id,
                expected_res_type="json",
                json_type="editing",
                temperature=temperature,
                max_tokens=self.max_tokens,
                client=self.client if isinstance(self.client, AsyncOpenAI) else None,
            )
        elif llm_provider == "claude":
            response_pyd_obj = await call_claude_api_async(
                prompt=prompt,
                model_id=self.model_id,
                expected_res_type="json",
                json_type="editing",
                temperature=temperature,
                max_tokens=self.max_tokens,
                client=self.client if isinstance(self.client, AsyncAnthropic) else None,
            )
        elif llm_provider == "llama3":
            response_pyd_obj = await call_llama3_async(
                prompt=prompt,
                expected_res_type="json",
                json_type="editing",
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Validate response
        self.validate_response(response_pyd_obj)
        return response_pyd_obj

    def validate_response(self, response_pyd_obj: BaseModel) -> None:
        """
        Validate the API response using Pydantic model validation.
        """
        try:
            # This will raise a ValidationError if the response does not match the model
            response_pyd_obj = EditingResponseModel(**response_pyd_obj.model_dump())
            logger.info("Pydantic model validation passed.")
        except ValidationError as e:
            logger.error(f"Pydantic model validation failed: {e}")
            raise ValueError(f"Invalid format: {e}")

    async def edit_for_dp_async(
        self,
        target_text: str,
        source_text: str,
        text_id: str = "",
        llm_provider: str = "",
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        *Async version of the method

        Re-edit the target text to better align w/t source text's dependency parsing (DP),
        leveraging the LLM API specified.

        Returns:
            dict: A dictionary in the format of {'text_id': "...", 'optimized_text': "..."}.
        """
        text_id = self.generate_text_id(text_id)
        prompt = self.format_prompt(STRUCTURE_TRANSFER_PROMPT, target_text, source_text)
        response_model = await self.call_llm_async(
            prompt,
            llm_provider=llm_provider if llm_provider else self.llm_provider,
            temperature=temperature,
        )

        # Extract the actual optimized content
        data = cast(EditingResponseModel, response_model)
        optimized_text = data.data.optimized_text
        # optimized_text = response_model.model_dump().get("optimized_text", "")
        result = {"text_id": text_id, "optimized_text": optimized_text}
        logger.info(f"Results updated: \n{result}")
        return result

        # response_dict = (
        #     response_model.data if isinstance(response_model, BaseModel) else {}
        # )
        # result = {"text_id": text_id, **response_dict}
        # logger.info(f"Results updated: \n{result}")
        # return result

    async def edit_for_entailment_async(
        self,
        premise_text: str,
        hypothesis_text: str,
        text_id: str = "",
        llm_provider: str = "",
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        *Async version of the method.

        Re-edit the target text to strengthen its entailment with the source text.
        Entailment is directional and its order is often the reverse of other comparisons':
        - Premise is to be transformed.
        - Hypothesis text serves as the reference and does not change.

        Example:
        Revise responsibility text to match with job description's requirement(s).

        Args:
            - premise (str): the text is to be transformed
            (the text that serves as the premise (e.g., a claim)).
            - hypothesis_text (str): The source text to "compare to" (not to be transformed)
            (the text that serves as the hypothesis (e.g., a follow-up statement).
            - text_id (str): Identifier of the target text (defaulted to None - unique IDs
            to be generated by UUID function) (i.e., the responsibility bullet text.)
            - llm_provider (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float): defaulted to 0.8 (a higher temperature setting is
            needed to give the model more flexibility/creativity).

            Note:
            - resp is short for responsibility
            - req is short for (job) requirement

        Returns:
            dict: A dictionary in the format of {'text_id': "...", 'optimized_text': "..."}.
        """
        text_id = self.generate_text_id(text_id)
        prompt = self.format_prompt(
            ENTAILMENT_ALIGNMENT_PROMPT, premise_text, hypothesis_text
        )
        response_model = await self.call_llm_async(
            prompt,
            llm_provider=llm_provider if llm_provider else self.llm_provider,
            temperature=temperature,
        )

        # Extract the actual optimized content
        data = cast(EditingResponseModel, response_model)
        optimized_text = data.data.optimized_text
        # optimized_text = response_model.model_dump().get("optimized_text", "")
        result = {"text_id": text_id, "optimized_text": optimized_text}
        logger.info(f"Results updated: \n{result}")
        return result

        # response_dict = (
        #     response_model.model_dump() if isinstance(response_model, BaseModel) else {}
        # )
        # result = {"text_id": text_id, **response_dict}
        # logger.info(f"Results updated: \n{result}")
        # return result

    async def edit_for_semantics_async(
        self,
        candidate_text: str,
        reference_text: str,
        text_id: str = "",
        llm_provider: str = "",
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        *Async version of the method.

        Re-edit the target text to better align w/t source text's semantics, leveraging LLMs.

        Example:
        Revise responsibility text to match with job description's requirement(s).

        Args:
            - candidate_text (str): The text to be transformed (i.e.,
            revised responsibility text).
            - reference_text (str): The text to be compared to (not transformed)
            to be modeled after (i.e., original responsibility text from resume).
            - text_id (str): Identifier of the target text (defaulted to None - unique IDs
            to be generated by UUID function) (i.e., the responsibility bullet text.)
            - llm_provider (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float): defaulted to 0.8 (a higher temperature setting is
            needed to give the model more flexibility/creativity).

            Note:
            - resp is short for responsibility
            - req is short for (job) requirement

        Returns:
            dict: A dictionary in the format of {'text_id': "...", 'optimized_text': "..."}.
        """
        text_id = self.generate_text_id(text_id)
        prompt = self.format_prompt(
            SEMANTIC_ALIGNMENT_PROMPT, candidate_text, reference_text
        )
        response_model = await self.call_llm_async(
            prompt,
            llm_provider=llm_provider if llm_provider else self.llm_provider,
            temperature=temperature,
        )

        # Extract the actual optimized content
        data = cast(EditingResponseModel, response_model)
        optimized_text = data.data.optimized_text

        logger.info(f"optimized_text: {optimized_text}")
        result = {"text_id": text_id, "optimized_text": optimized_text}
        logger.info(f"Results updated: \n{result}")
        return result

        # response_dict = (
        #     response_model.model_dump() if isinstance(response_model, BaseModel) else {}
        # )
        # result = {"text_id": text_id, **response_dict}
        # logger.info(f"Results updated: \n{result}")
        # return result

    async def edit_for_semantics_and_entailment_async(
        self,
        candidate_text: str,
        reference_text: str,
        text_id: str = "",
        llm_provider: str = "",
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        *Async version of the method.

        Re-edit the target text to better align w/t source text's semantics and strengthen
        the entailment relationships between the two texts, leveraging LLMs.

        Args:
            - candidate_text (str): Original text to be transformed by the model
            (i.e., the riginal responsibility text to be revised.)
            - reference_text (str): Text that the candidate text is being compared to
            (i.e., requirement text to optimize against.)
            - text_id (int): (Optional) Identifier for the responsibility text.
            Default to None (unique ids to be generated with UUID function)
            - llm_provider (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float, optional): Temperature setting for this specific call.

        Returns:
            dict: Contains 'text_id' and 'optimized_text' after revision.
        """
        text_id = self.generate_text_id(text_id)
        prompt = self.format_prompt(
            SEMANTIC_ENTAILMENT_ALIGNMENT_PROMPT, candidate_text, reference_text
        )
        response_model = await self.call_llm_async(
            prompt,
            llm_provider=llm_provider if llm_provider else self.llm_provider,
            temperature=temperature,
        )

        # Extract the actual optimized content
        data = cast(EditingResponseModel, response_model)
        optimized_text = data.data.optimized_text
        result = {"text_id": text_id, "optimized_text": optimized_text}
        logger.info(f"Results updated: \n{result}")
        return result

        # response_dict = (
        #     response_model.model_dump() if isinstance(response_model, BaseModel) else {}
        # )
        # result = {"text_id": text_id, **response_dict}
        # logger.info(f"Results updated: \n{result}")
        # return result
