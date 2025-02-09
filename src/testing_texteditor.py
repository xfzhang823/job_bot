from pathlib import Path
import logging
import logging_config
from anthropic import Anthropic
from openai import OpenAI

from evaluation_optimization.resume_editor import TextEditor
from llm_providers.llm_api_utils import get_claude_api_key, get_openai_api_key
from project_config import CLAUDE_HAIKU

from project_config import CLAUDE_HAIKU, GPT_4_TURBO

# Set up logging
logger = logging.getLogger(__name__)

text_to_modify = "Our cloud-based platform offers advanced workload optimization by dynamically redistributing tasks across your organization’s teams. With built-in analytics and predictive models, you can better allocate resources, improve turnaround times, and maintain a more balanced workload distribution across departments."

text_to_compare = "We are looking for a solution to streamline internal operations by ensuring optimal resource utilization across teams, reducing bottlenecks, and achieving consistent delivery times without overburdening employees."


def main_claude():
    # Instantiate the client
    api_key = get_claude_api_key()
    client = OpenAI(api_key=api_key)

    # Initialize TextEditor
    text_editor = TextEditor(
        llm_provider="claude",
        model_id=CLAUDE_HAIKU,
        client=client,
        max_tokens=1024,
    )

    print(f"Initialized TextEditor with llm_provider={text_editor.llm_provider}")

    print(
        f"Original:\nText to modify: {text_to_modify}\nText to Compare: {text_to_compare}"
    )

    # Step 1: Align Semantic
    try:
        logger.debug("Starting semantic alignment step...")
        revised = text_editor.edit_for_semantics(
            candidate_text=text_to_modify,
            reference_text=text_to_compare,
            temperature=0.5,
        )
        revised_text_1 = revised.data.optimized_text
        print(f"More Similar in Meaning: \n{revised_text_1}")
    except Exception as e:
        logger.error(f"Error in semantic alignment step: {e}")
        return

    # Step 2: Align Entailment
    try:
        logger.debug("Starting entailment alignment step...")
        revised = text_editor.edit_for_entailment(
            premise_text=revised_text_1,
            hypothesis_text=text_to_compare,
            temperature=0.6,
        )
        revised_text_2 = revised.data.optimized_text
        print(f"Strengthen Inference: \n{revised_text_2}")
    except Exception as e:
        logger.error(f"Error in entailment alignment step: {e}")
        return

    # Step 3: Align Original Sentence's Dependency Parsing (DP)
    try:
        logger.debug("Starting dependency parsing alignment step...")
        revised = text_editor.edit_for_dp(
            target_text=revised_text_2,
            source_text=text_to_modify,
            temperature=0.9,
        )
        revised_text_3 = revised.data.optimized_text
        print(f"Make Authentic: \n{revised_text_3}")
    except Exception as e:
        logger.error(f"Error in dependency parsing alignment step: {e}")
        return


def main_gpt():
    # Instantiate the client
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)

    # Initialize TextEditor
    text_editor = TextEditor(
        llm_provider="openai",
        model_id=GPT_4_TURBO,
        client=client,
        max_tokens=1024,
    )

    logger.debug(f"Initialized TextEditor with llm_provider={text_editor.llm_provider}")

    print(
        f"Original:\nText to modify: {text_to_modify}\nText to Compare: {text_to_compare}"
    )

    # Step 1: Align Semantic
    try:
        logger.debug("Starting semantic alignment step...")
        revised = text_editor.edit_for_semantics(
            candidate_text=text_to_modify,
            reference_text=text_to_compare,
            temperature=0.5,
        )
        revised_text_1 = revised.data.optimized_text
        print(f"More Similar in Meaning: \n{revised_text_1}")
    except Exception as e:
        logger.error(f"Error in semantic alignment step: {e}")
        return

    # Step 2: Align Entailment
    try:
        logger.debug("Starting entailment alignment step...")
        revised = text_editor.edit_for_entailment(
            premise_text=revised_text_1,
            hypothesis_text=text_to_compare,
            temperature=0.6,
        )
        revised_text_2 = revised.data.optimized_text
        print(f"Strengthen Inference: \n{revised_text_2}")
    except Exception as e:
        logger.error(f"Error in entailment alignment step: {e}")
        return

    # Step 3: Align Original Sentence's Dependency Parsing (DP)
    try:
        logger.debug("Starting dependency parsing alignment step...")
        revised = text_editor.edit_for_dp(
            target_text=revised_text_2,
            source_text=text_to_modify,
            temperature=0.9,
        )
        revised_text_3 = revised.data.optimized_text
        print(f"Make Authentic: \n{revised_text_3}")
    except Exception as e:
        logger.error(f"Error in dependency parsing alignment step: {e}")
        return


if __name__ == "__main__":
    main_gpt()
    # main_claude()
