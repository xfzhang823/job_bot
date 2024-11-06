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
import logging_config
from pydantic import ValidationError, BaseModel
from openai import OpenAI
from anthropic import Anthropic

from prompts.prompt_templates import (
    SEMANTIC_ALIGNMENT_PROMPT,
    ENTAILMENT_ALIGNMENT_PROMPT,
    SEMANTIC_ENTAILMENT_ALIGNMENT_PROMPT,
    STRUCTURE_TRANSFER_PROMPT,
)
from utils.llm_api_utils import (
    get_openai_api_key,
    get_claude_api_key,
    call_claude_api,
    call_openai_api,
    call_llama3,
)
from utils.validation_utils import validate_json_response
from models.llm_response_models import (
    EditingResponseModel,
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponseModel,
    JobSiteResponseModel,
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
        - edit_for_dp(target_text, source_text, text_id=None, model=None, temperature=None):
            Edits text to align with source text's dependency parsing (DP).
        - edit_for_entailment(target_text, source_text, text_id=None, model=None, temperature=None):
            Edits text based on semantic entailment.
        - edit_for_semantics(target_text, source_text, text_id=None, model=None, temperature=None):
            Edits text to align with source text's semantics.
        - edit_for_semantics_and_entailment(candidate_text, reference_text, text_id=None, model=None,
        temperature=None):
            Edits text to align with source text's semantics and strengthen entailment relationships.
    """

    def __init__(
        self,
        model: str = "openai",
        model_id: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1056,
        client: Optional[Union[OpenAI, Anthropic]] = None,
    ):
        self.model = model  # Default model ('openai')
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client  # Store the passed client

        # Conditionally initialize the client based on model
        if self.model == "claude" and not self.client:
            claude_api_key = get_claude_api_key()
            self.client = Anthropic(api_key=claude_api_key)
        if self.model == "openai" and not self.client:
            # Initialize OpenAI API client if it's not provided
            openai_api_key = get_openai_api_key()  # Fetch the API key
            self.client = OpenAI(
                api_key=openai_api_key
            )  # Instantiate OpenAI API client
        elif self.model == "llama3":
            # If using LLaMA3, no OpenAI client initialization is needed
            # You may initialize LLaMA3-specific settings here if needed
            logger.info("Using LLaMA3 model for text editing.")
        elif not self.client:
            raise ValueError(f"Unsupported model: {self.model}")

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
        self, prompt: str, model: str = "openai", temperature: Optional[float] = None
    ) -> Union[
        str,
        JSONResponse,
        TabularResponse,
        TextResponse,
        CodeResponse,
        EditingResponseModel,
        JobSiteResponseModel,
    ]:
        """
        Call the specified LLM API (OpenAI or LLaMA3) with the provided prompt and return the validated response.

        Args:
            prompt (str): The formatted prompt to send to the LLM API.
            model (str): The model to use for the API call ('openai' or 'llama3').
            temperature (float, optional): Temperature setting for this specific API call.
                                        If None, uses the class-level temperature.

        Returns:
            dict: Validated response data as a dictionary, structured according to the JSON schema.

        Raises:
            ValueError: If the model type is unsupported or if the response does not pass validation.
        """
        # Set temperature to class default if not provided
        temperature = temperature if temperature is not None else self.temperature

        # Select the appropriate LLM API call
        if model == "openai":
            response_pyd_obj = call_openai_api(
                prompt=prompt,
                model_id=self.model_id,
                expected_res_type="json",
                json_type="editing",
                temperature=temperature,
                max_tokens=self.max_tokens,
                client=self.client if isinstance(self.client, OpenAI) else None,
            )

        if model == "claude":
            response_pyd_obj = call_claude_api(
                prompt=prompt,
                model_id=self.model_id,
                expected_res_type="json",
                json_type="editing",
                temperature=temperature,
                max_tokens=self.max_tokens,
                client=self.client if isinstance(self.client, Anthropic) else None,
            )

        elif model == "llama3":
            response_pyd_obj = call_llama3(
                prompt=prompt,
                expected_res_type="json",
                json_type="editing",
                temperature=temperature,
                max_tokens=self.max_tokens,
            )

        else:
            raise ValueError(f"Unsupported model: {model}")

        # Validate the response
        self.validate_response(response_pyd_obj)
        # Return the validated response dictionary
        return response_pyd_obj

    def validate_response(self, response_pyd_obj: Union[BaseModel, dict]) -> None:
        """
        Validate the API response (Pydantic model or dictionary) using JSON Schema.

        Args:
            response_pyd_obj (Union[BaseModel, dict]): The response data, either as a Pydantic model or a dictionary.

        Return: None

        Raises:
            ValueError: If the response does not match the expected JSON schema.
        """
        # Convert Pydantic model to dictionary if needed
        response_dict = (
            response_pyd_obj.model_dump()
            if isinstance(response_pyd_obj, BaseModel)
            else response_pyd_obj
        )

        try:
            # Perform JSON schema validation
            jsonschema.validate(instance=response_dict, schema=LLM_RES_JSON_SCHEMA)
            logger.info("JSON schema validation passed.")
        except ValidationError as e:
            logger.error(f"JSON schema validation failed: {e}")
            raise ValueError(f"Invalid JSON format: {e}")

    def edit_for_dp(
        self,
        target_text: str,
        source_text: str,
        text_id: str = "",
        model: str = "",
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        Re-edit the target text to better align w/t source text's dependency parsing (DP),
        leveraging the OpenAI API.

        Example:
        Re-edit revised responsibility to match with the original responsibility text's DP
        to perserve the tone & style.

        Args:
            - target_text (str): The target text to be transformed (i.e.,
            revised responsibility text).
            - source_text (str): The source text from whose "dependency parsing"
            to be modeled after (i.e., original responsibility text from resume).
            - model_id (str): OpenAI model to use (default is 'gpt-4').
            - text_id (str): Identifier of the target text (defaulted to None - unique IDs
            to be generated by UUID function) (i.e., the responsibility bullet text.)
            - model (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float): defaulted to 0.8 (a higher temperature setting is
            needed to give the model more flexibility/creativity).

            Note:
            - resp is short for responsibility
            - req is short for (job) requirement

        Returns:
            dict: A dictionary in the format of {'text_id': "...", 'optimized_text': "..."}.
        """
        # Generate text id (as unique identifier)
        text_id = self.generate_text_id(text_id)
        prompt = self.format_prompt(STRUCTURE_TRANSFER_PROMPT, target_text, source_text)

        # Call call_llm method to fetch a LLM response (in the form of a pyd model)
        response_model = self.call_llm(
            prompt, model=model if model else self.model, temperature=temperature
        )

        # pyd model -> dictionary
        if isinstance(response_model, BaseModel):
            response_dict = response_model.model_dump()
        else:
            raise ValueError(
                "The response model is not a Pydantic model and does not have 'model_dumpt"
            )

        # Combine text_id and response_dict to form a new dictionary
        result = {"text_id": text_id, **response_dict}
        logger.info(f"Results updated: \n{result}")
        return result

    def edit_for_entailment(
        self,
        premise_text: str,
        hypothesis_text: str,
        text_id: str = "",
        model: str = "",
        temperature: Optional[float] = None,
    ) -> Dict:
        """
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
            - model (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float): defaulted to 0.8 (a higher temperature setting is
            needed to give the model more flexibility/creativity).

            Note:
            - resp is short for responsibility
            - req is short for (job) requirement

        Returns:
            dict: A dictionary in the format of {'text_id': "...", 'optimized_text': "..."}.
        """
        # Generate text id (as unique identifier)
        text_id = self.generate_text_id(text_id)
        prompt = self.format_prompt(
            ENTAILMENT_ALIGNMENT_PROMPT, premise_text, hypothesis_text
        )

        # Call call_llm method to fetch a LLM response (in the form of a pyd model)
        response_model = self.call_llm(
            prompt, model=model if model else self.model, temperature=temperature
        )

        # pyd model -> dictionary
        if isinstance(response_model, BaseModel):
            response_dict = response_model.model_dump()
        else:
            raise ValueError(
                "The response model is not a Pydantic model and does not have 'model_dumpt"
            )
        # Combine text_id and response_dict to form a new dictionary
        result = {"text_id": text_id, **response_dict}
        logger.info(f"Results updated: \n{result}")
        return result

    def edit_for_semantics(
        self,
        candidate_text: str,
        reference_text: str,
        text_id: str = "",
        model: str = "",
        temperature: Optional[float] = None,
    ) -> Dict:
        """
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
            - model (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float): defaulted to 0.8 (a higher temperature setting is
            needed to give the model more flexibility/creativity).

            Note:
            - resp is short for responsibility
            - req is short for (job) requirement

        Returns:
            dict: A dictionary in the format of {'text_id': "...", 'optimized_text': "..."}.
        """
        # Generate text id (as unique identifier)
        text_id = self.generate_text_id(text_id)
        prompt = self.format_prompt(
            SEMANTIC_ALIGNMENT_PROMPT, candidate_text, reference_text
        )

        # Call call_llm method to fetch a LLM response (in the form of a pyd model)
        response_model = self.call_llm(
            prompt,
            model=model if model else self.model,
            temperature=temperature,
        )

        # pyd model -> dictionary
        if isinstance(response_model, BaseModel):
            response_dict = response_model.model_dump()
        else:
            raise ValueError(
                "The response model is not a Pydantic model and does not have 'model_dumpt"
            )

        # Combine text_id and response_dict to form a new dictionary
        result = {"text_id": text_id, **response_dict}
        logger.info(f"Results updated: \n{result}")
        return result

    def edit_for_semantics_and_entailment(
        self,
        candidate_text: str,
        reference_text: str,
        text_id: str = "",
        model: str = "",
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        Re-edit the target text to better align w/t source text's semantics and strengthen
        the entailment relationships between the two texts, leveraging LLMs.

        Args:
            - candidate_text (str): Original text to be transformed by the model
            (i.e., the riginal responsibility text to be revised.)
            - reference_text (str): Text that the candidate text is being compared to
            (i.e., requirement text to optimize against.)
            - text_id (int): (Optional) Identifier for the responsibility text.
            Default to None (unique ids to be generated with UUID function)
            - model (str, optional): The model to use for the API call ('openai' or 'llama3').
            - temperature (float, optional): Temperature setting for this specific call.

        Returns:
            dict: Contains 'text_id' and 'optimized_text' after revision.
        """
        # Generate text id (as unique identifier)
        text_id = self.generate_text_id(text_id)
        prompt = self.format_prompt(
            SEMANTIC_ENTAILMENT_ALIGNMENT_PROMPT, candidate_text, reference_text
        )

        # Call call_llm method to fetch a LLM response (in the form of a pyd model)
        response_model = self.call_llm(
            prompt,
            model=model if model else self.model,
            temperature=temperature,
        )

        # pyd model -> dictionary
        if isinstance(response_model, BaseModel):
            response_dict = response_model.model_dump()
        else:
            raise ValueError(
                "The response model is not a Pydantic model and does not have 'model_dumpt"
            )

            # Combine text_id and response_dict to form a new dictionary
        result = {"text_id": text_id, **response_dict}
        logger.info(f"Results updated: \n{result}")
        return result
