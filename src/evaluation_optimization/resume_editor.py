"""
File: resume_editing.py
Author: Xiao-Fei Zhang
Last updated: 2024 Sep 12
"""

import os
import logging
import uuid
import json
import openai
import jsonschema
import pandas as pd
from typing import Dict, Optional, Union
from pydantic import ValidationError, BaseModel
from openai import OpenAI
from anthropic import Anthropic

from prompts.prompt_templates import (
    SEMANTIC_ALIGNMENT_PROMPT,
    ENTAILMENT_ALIGNMENT_PROMPT,
    SEMANTIC_ENTAILMENT_ALIGNMENT_PROMPT,
    STRUCTURE_TRANSFER_PROMPT,
)
from llm_providers.llm_api_utils import (
    get_openai_api_key,
    get_anthropic_api_key,
    call_anthropic_api,
    call_openai_api,
    call_llama3,
)
from utils.validation_utils import validate_json_response
from models.llm_response_models import (
    EditingResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponse,
    JobSiteResponse,
    NestedRequirements,
    RequirementsResponse,
)
from project_config import ANTHROPIC, OPENAI, GPT_35_TURBO, GPT_4_TURBO
import logging_config


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


class TextEditor:
    """
    A class to edit and optimize a text, given another text to "match" to, using language models
    like OpenAI GPT-4 or LLaMA3.

    This class provides methods to perform text editing based on semantic entailment
    and dependency parsing alignment using various language models via API calls.

    Attributes:
        model (str): The model to use for the API calls ('openai' or 'llama3').
        client (any): instantiated API for OpenAI API calls.
        model_id (str): The model ID to use for the API calls (e.g., 'gpt-4-turbo').
        temperature (float): Temperature parameter for the model, affecting response variability.
        max_tokens (int): The maximum number of tokens to generate for the model output.

    Methods:
        - generate_text_id(text_id=None): Generates a unique text ID using UUID.
        - format_prompt(prompt_template, content_1, content_2):
            Formats the prompt using a provided template.
        - call_llm(prompt, model=None, temperature=None):
            Calls the specified LLM API (OpenAI or LLaMA3) with the given prompt and returns
            the response.
        - validate_response(response_dict):
            Validates a response dictionary against a predefined JSON schema.
        - edit_for_dp(target_text, source_text, text_id=None, model_id=None, temperature=None):
            Edits text to align with source text's dependency parsing (DP).
        - edit_for_entailment(target_text, source_text, text_id=None, model_id=None, temperature=None):
            Edits text based on semantic entailment.
        - edit_for_semantics(target_text, source_text, text_id=None, model_id=None, temperature=None):
            Edits text to align with source text's semantics.
        - edit_for_semantics_and_entailment(candidate_text, reference_text, text_id=None, model_id=None,
        temperature=None):
            Edits text to align with source text's semantics and strengthen entailment relationships.
    """

    def __init__(
        self,
        llm_provider: str = OPENAI,
        model_id: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1056,
        client: Optional[Union[OpenAI, Anthropic]] = None,
    ):
        self.llm_provider = llm_provider  # Default model ('openai')
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client  # Store the passed client

        logger.debug(f"Client initialized: {self.client}")  # debugging

        # Conditionally initialize the client based on model
        if self.llm_provider.lower() == "anthropic" and not self.client:
            claude_api_key = get_anthropic_api_key()
            self.client = Anthropic(api_key=claude_api_key)
        if self.llm_provider.lower() == "openai" and not self.client:
            # Initialize OpenAI API client if it's not provided
            openai_api_key = get_openai_api_key()  # Fetch the API key
            self.client = OpenAI(
                api_key=openai_api_key
            )  # Instantiate OpenAI API client
        elif self.llm_provider.lower() == "llama3":
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

    def call_llm(
        self,
        prompt: str,
        llm_provider: str = "openai",
        temperature: Optional[float] = None,
    ) -> Union[
        JSONResponse,
        TabularResponse,
        TextResponse,
        CodeResponse,
        EditingResponse,
        JobSiteResponse,
        NestedRequirements,
        RequirementsResponse,
    ]:
        """
        Call the specified LLM API (OpenAI, Claude, LLaMA3) with the provided prompt
        and return the validated response.

        Args:
            - prompt (str): The formatted prompt to send to the LLM API.
            - llm_provider (str): The provider to use for the API call
            ('openai', 'claude', 'llama3').
            - temperature (float, optional):
                Temperature setting for this specific API call.
                Defaults to the class-level temperature if not provided.

        Returns:
            Union[str, JSONResponse, TabularResponse, TextResponse, CodeResponse,
            EditingResponseModel]:
                The validated response data as a dictionary or structured Pydantic model.

        Raises:
            ValueError: If the model type is unsupported or the response
            does not pass validation.
        """
        logger.debug(f"Entering call_llm with llm_provider={llm_provider}")

        # Assert that llm_provider is valid
        assert llm_provider in [
            "openai",
            "claude",
            "llama3",
        ], f"Unexpected llm_provider: {llm_provider}"

        # Set temperature to class default if not provided
        temperature = temperature if temperature is not None else self.temperature

        # Fixed json_type for all methods in TextEditor
        json_type = "editing"

        logger.debug(f"LLM provider passed to call_llm: {llm_provider}")

        # Select the appropriate LLM API call
        if llm_provider == "openai":
            logger.debug("Using OpenAI client.")  # debugging

            response_model = call_openai_api(
                prompt=prompt,
                model_id=self.model_id,
                expected_res_type="json",
                json_type=json_type,  # Fixed json_type
                temperature=temperature,
                max_tokens=self.max_tokens,
                client=self.client if isinstance(self.client, OpenAI) else None,
            )

            logger.debug(
                f"Response received from call_openai_api: {type(response_model).__name__}"
            )  # TODO debugging; delete later
            logger.info(f"returned model: {response_model}")

        if llm_provider.lower() == "anthropic":
            logger.debug("Using Anthropic client.")  # debugging

            response_model = call_anthropic_api(
                prompt=prompt,
                model_id=self.model_id,
                expected_res_type="json",
                json_type=json_type,  # Fixed json_type
                temperature=temperature,
                max_tokens=self.max_tokens,
                client=self.client if isinstance(self.client, Anthropic) else None,
            )

        elif llm_provider == "llama3":
            logger.debug("Using LLaMA3.")

            response_model = call_llama3(
                prompt=prompt,
                expected_res_type="json",
                json_type=json_type,  # Fixed json_type
                temperature=temperature,
                max_tokens=self.max_tokens,
            )

        else:
            logger.error(f"Unsupported model: {llm_provider}")
            raise ValueError(f"Unsupported model: {llm_provider}")

        return response_model

    def edit_for_dp(
        self,
        target_text: str,
        source_text: str,
        temperature: Optional[float] = None,
    ) -> EditingResponse:
        """
        Re-edit the target text to better align w/t source text's dependency parsing (DP),
        leveraging the OpenAI API.

        Example:
        Re-edit revised responsibility to match with the original responsibility text's DP
        to preserve the tone & style.

        Args:
            - target_text (str): The target text to be transformed (i.e.,
            revised responsibility text).
            - source_text (str): The source text from whose "dependency parsing"
            to be modeled after (i.e., original responsibility text from resume).
            - model (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float, optional): Defaults to 0.8 (a higher temperature setting
            is needed to give the model more flexibility/creativity).

            Note:
            - resp is short for responsibility.
            - req is short for (job) requirement.

        Returns:
            EditingResponseModel: A Pydantic model containing the optimized text.
        """
        # Generate a unique text id (preserved for potential future use)
        text_id = f"{hash(target_text)}_{hash(source_text)}"
        prompt = self.format_prompt(STRUCTURE_TRANSFER_PROMPT, target_text, source_text)

        # Call call_llm method to fetch a LLM response (in the form of a Pydantic model)
        response_model = self.call_llm(
            prompt,
            llm_provider=self.llm_provider,
            temperature=temperature,
        )

        # Ensure the response is an instance of EditingResponseModel
        if not isinstance(response_model, EditingResponse):
            raise ValueError("The response is not an instance of EditingResponseModel.")

        # Log and return the response model
        logger.info(
            f"Dependency Parsing Alignment Result (text_id={text_id}): {response_model}"
        )
        return response_model

    def edit_for_entailment(
        self,
        premise_text: str,
        hypothesis_text: str,
        temperature: Optional[float] = None,
    ) -> EditingResponse:
        """
        Re-edit the target text to strengthen its entailment with the source text.
        Entailment is directional and its order is often the reverse of other comparisons:
        - Premise is to be transformed.
        - Hypothesis text serves as the reference and does not change.

        Example:
        Revise responsibility text to match with job description's requirement(s).

        Args:
            - premise_text (str): The text to be transformed (the premise, e.g., a claim).
            - hypothesis_text (str): The source text to "compare to" (not to be transformed,
            e.g., a follow-up statement).
            - model (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float, optional): Defaults to 0.8 (a higher temperature setting
            is needed to give the model more flexibility/creativity).

            Note:
            - resp is short for responsibility.
            - req is short for (job) requirement.

        Returns:
            EditingResponseModel: A Pydantic model containing the optimized text.
        """
        # Generate a unique text id (preserved for potential future use)
        text_id = f"{hash(premise_text)}_{hash(hypothesis_text)}"
        prompt = self.format_prompt(
            ENTAILMENT_ALIGNMENT_PROMPT, premise_text, hypothesis_text
        )

        # Call call_llm method to fetch a LLM response (in the form of a Pydantic model)
        response_model = self.call_llm(
            prompt,
            llm_provider=self.llm_provider,
            temperature=temperature,
        )

        # Ensure the response is an instance of EditingResponseModel
        if not isinstance(response_model, EditingResponse):
            raise ValueError("The response is not an instance of EditingResponseModel.")

        # Log and return the response model
        logger.info(
            f"Entailment Alignment Result (text_id={text_id}): {response_model}"
        )
        return response_model

    def edit_for_semantics(
        self,
        candidate_text: str,
        reference_text: str,
        temperature: Optional[float] = None,
    ) -> EditingResponse:
        """
        Re-edit the target text to better align w/t source text's semantics, leveraging LLMs.

        Example:
        Revise responsibility text to match with job description's requirement(s).

        Args:
            - candidate_text (str): The text to be transformed (i.e.,
            revised responsibility text).
            - reference_text (str): The text to be compared to (not transformed,
            e.g., original responsibility text from resume).
            - model (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float, optional): Defaults to 0.8 (a higher temperature setting
            is needed to give the model more flexibility/creativity).

            Note:
            - resp is short for responsibility.
            - req is short for (job) requirement.

        Returns:
            EditingResponseModel: A Pydantic model containing the optimized text.
        """
        # Generate a unique text id (preserved for potential future use)
        text_id = f"{hash(candidate_text)}_{hash(reference_text)}"
        prompt = self.format_prompt(
            SEMANTIC_ALIGNMENT_PROMPT, candidate_text, reference_text
        )

        # Call call_llm method to fetch a LLM response (in the form of a Pydantic model)
        response_model = self.call_llm(
            prompt,
            llm_provider=self.llm_provider,
            temperature=temperature,
        )

        # Debugging
        logger.debug(
            f"Response model type in edit_for_semantics: {type(response_model).__name__}"
        )

        # Ensure the response is an instance of EditingResponseModel
        if not isinstance(response_model, EditingResponse):
            raise ValueError("The response is not an instance of EditingResponseModel.")

        # Log and return the response model
        logger.info(f"Semantic Alignment Result (text_id={text_id}): {response_model}")
        return response_model
