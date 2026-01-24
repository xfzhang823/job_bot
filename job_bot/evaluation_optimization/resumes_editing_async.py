"""
Filename: resume_editing_async.py
Lst updated on: 2024 Oct 21

Not tested/debugged yet
"""

# Import from standard & 3rd party
import logging
from typing import Dict, Tuple, Optional, Set
import asyncio
from pydantic import ValidationError


# User defined
from job_bot.evaluation_optimization.resume_editor_async import TextEditorAsync
from job_bot.models.resume_job_description_io_models import (
    OptimizedText,
    ResponsibilityMatch,
    ResponsibilityMatches,
)
from job_bot.config.project_config import OPENAI, ANTHROPIC, GPT_4_1_NANO, CLAUDE_HAIKU

# Set up logger
logger = logging.getLogger(__name__)


EligibleMap = Dict[str, Set[str]]


async def modify_resp_based_on_reqs_async(
    resp_key: str, resp: str, reqs: Dict[str, str], llm_provider: str, model_id: str
) -> Tuple[str, ResponsibilityMatch]:
    """
    *This is the async version of the modify_resps_based_on_reqs function.

    Modify a single responsibility text by aligning it with multiple job requirements.

    This function modifies one responsibility (`resp`) by matching it against multiple
    job requirements (`reqs`). The alignment process involves three stages:
    semantic alignment, entailment alignment, and dependency parsing (DP) alignment.
    The TextEditor class is used for these modifications based on the provided model.

    The process includes:
    1. Semantic Alignment: Adjusting the responsibility text to ensure it semantically
       aligns with the job requirement.
    2. Entailment Alignment: Ensuring that the responsibility text can be logically
       inferred from the job requirement.
    3. Dependency Parsing Alignment (DP): Refining the final responsibility text
       while maintaining its original structure as much as possible.

    Args:
        resp_key (str): Unique identifier for the responsibility.
        resp (str): The responsibility text to be modified.
        reqs (dict): A dictionary of job requirements, where keys are unique requirement
            identifiers and values are the requirement texts.
        model (str): Name of the model to be used (e.g., "openai").
        model_id (str): The specific model version to be used (e.g., "gpt-3.5-turbo").

    Returns:
        tuple: A tuple containing:
            - 'resp_key' (str): The same responsibility_key passed to the function.
            - 'local_modifications' (dict): A dictionary of modified responsibility texts
              for each requirement, keyed by the requirement identifier.
              The value is a dict containing the final `optimized_text`.

    Example:
        >>> await modify_resp_based_on_reqs_async(
                resp_key="resp1",
                resp="Managed a team of 5 developers",
                reqs={"req1": "Experience leading software development teams."},
                llm_provider="openai",
                model_id="gpt-4-turbo"
            )

    !Note: Rationale for Temperature Settings in Text Alignment:

        *1. Semantic Alignment (Temperature ≈ 0.45):
        - Moderate temperature allows the model to understand and lightly reshape
        the candidate text to improve semantic proximity to the job requirement,
        while avoiding excessive creativity or résumé-style embellishment.
        This stage focuses on semantic understanding rather than stylistic rewriting.

        *2. Entailment Alignment (Temperature ≈ 0.30):
        - Slightly lower temperature constrains the model to strengthen the logical
        support between the premise (candidate text) and the hypothesis
        (job requirement) without introducing new claims, audiences, or outcomes.
        This step tightens entailment while limiting abstract or inflated language.

        *3. Dependency Parsing Alignment / Re-Anchoring (Temperature ≈ 0.10–0.20):
        - Low temperature enforces deterministic, source-anchored edits that
        reapply the original candidate text’s structure and tone, and explicitly
        remove or soften any unsupported claims introduced earlier.
        This step prioritizes factual fidelity and authenticity over expressive
        rewriting.

        Overall, the temperature strategy intentionally decreases across stages as
        authority shifts from requirement-driven alignment toward the original resume.
        Early stages allow limited exploration to establish semantic and logical
        fit, while the final stage enforces truth anchoring, deletion of hallucinated
        content, and preservation of the candidate’s authentic voice and scope.

    """

    # Initialize the client based on llm_provider if needed
    if llm_provider == OPENAI:
        client = None  # * don't instantiate here; use global client in lower-level
        logger.info("OpenAI API initialized.")

    elif llm_provider == ANTHROPIC:
        client = None  # * don't instantiate here; use global client in lower-level
        logger.info("Claude API initialized.")

    elif llm_provider == "llama3":
        client = None  # No client needed for local Llama3
        logger.info("Using local Llama3 model.")

    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # Initialize the async text editor with the async client
    text_editor = TextEditorAsync(
        llm_provider=llm_provider, model_id=model_id, client=client, max_tokens=1024
    )

    local_modifications = {}

    try:
        for req_key, req in reqs.items():
            logger.info(f"Modifying responsibility: {resp} \nwith requirement: {req}")

            # Step 1: Semantic Alignment
            revised = await text_editor.edit_for_semantics_async(
                candidate_text=resp, reference_text=req, temperature=0.45
            )  # Set temperature low to make the change light
            revised_text_1 = revised.data.optimized_text

            # Step 2: Entailment Alignment
            revised = await text_editor.edit_for_entailment_async(
                premise_text=revised_text_1, hypothesis_text=req, temperature=0.3
            )  # set temp low to make the change moderate
            revised_text_2 = revised.data.optimized_text

            # Step 3: Dependency Parsing Alignment -> Original Text
            revised = await text_editor.edit_for_dp_async(
                target_text=revised_text_2, source_text=resp, temperature=0.15
            )  # set temp high to make large change to retain original text's structure and tone
            revised_text_3 = revised.data.optimized_text

            # Store the optimized text
            optimized_text = OptimizedText(optimized_text=revised_text_3)
            local_modifications[req_key] = optimized_text

        # Wrap the modifications under optimized_by_requirements
        validated_modifications = ResponsibilityMatch(
            optimized_by_requirements=local_modifications
        )

    except Exception as e:
        logger.error(f"Failed to modify responsibility {resp_key}: {e}")
        # Fallback for error cases
        local_modifications[req_key] = OptimizedText(optimized_text=resp)
        validated_modifications = ResponsibilityMatch(
            optimized_by_requirements=local_modifications
        )

    return resp_key, validated_modifications


async def modify_multi_resps_based_on_reqs_async(
    responsibilities: Dict[str, str],
    requirements: Dict[str, str],
    llm_provider: str,
    model_id: str,
    no_of_concurrent_workers: int = 5,
    eligible_map: Optional[EligibleMap] = None,
) -> ResponsibilityMatches:
    """
    Async batch editor for aligning responsibilities to job requirements.

    This function edits multiple responsibilities by aligning each responsibility
    to one or more job requirements using an LLM-driven, three-stage process:
    1) Semantic alignment
    2) Entailment alignment
    3) Dependency-structure preservation

    The function supports both dense and pair-filtered execution modes.

    Behavior
    --------
    - Default (eligible_map=None):
        Each responsibility is edited against the full `requirements` dictionary.

    - Filtered (eligible_map provided):
        Each responsibility is edited only against its allowed requirement keys
        (`eligible_map[resp_key]`). Responsibilities with no eligible requirements
        are skipped (no LLM call) and return an empty match.

    Concurrency
    -----------
    Per-responsibility edit tasks are executed concurrently and bounded by an
    asyncio.Semaphore to limit the number of in-flight LLM calls.

    Args:
        responsibilities:
            Mapping of responsibility_key -> responsibility text.
        requirements:
            Mapping of requirement_key -> requirement text.
        llm_provider:
            LLM provider identifier (e.g., "openai", "anthropic").
        model_id:
            Model identifier used for editing (e.g., "gpt-4.1-nano").
        no_of_concurrent_workers:
            Maximum number of concurrent responsibility-level edit tasks.
        eligible_map:
            Optional mapping of responsibility_key -> set(requirement_key)
            enabling per-responsibility pair filtering. When provided,
            only eligible pairs are sent to the LLM.

    Returns:
        ResponsibilityMatches:
            Pydantic model mapping responsibility_key ->
            ResponsibilityMatch(optimized_by_requirements={...}),
            containing edited responsibility texts keyed by requirement.
    """

    semaphore = asyncio.Semaphore(no_of_concurrent_workers)

    async def modify_resp_with_limit(resp_key: str, resp: str):
        # Decide which requirements this responsibility is allowed to use
        if eligible_map is None:
            reqs_sub = requirements
        else:
            keep = eligible_map.get(resp_key, set())
            if not keep:
                logger.info("⏭️ SKIP %s (no eligible requirements)", resp_key)
                return (resp_key, ResponsibilityMatch(optimized_by_requirements={}))

            reqs_sub = {k: requirements[k] for k in keep if k in requirements}
            if not reqs_sub:
                logger.info("⏭️ SKIP %s (eligible req keys missing)", resp_key)
                return (resp_key, ResponsibilityMatch(optimized_by_requirements={}))

        async with semaphore:
            logger.info("🔄 START %s | reqs=%d", resp_key, len(reqs_sub))
            result = await modify_resp_based_on_reqs_async(
                resp_key, resp, reqs_sub, llm_provider, model_id
            )
            logger.info(
                "✅ DONE %s | optimized=%d",
                resp_key,
                len(result[1].optimized_by_requirements),
            )
            return result

    items = (
        [
            (rk, responsibilities[rk])
            for rk in eligible_map.keys()
            if rk in responsibilities
        ]
        if eligible_map is not None
        else list(responsibilities.items())
    )

    results = await asyncio.gather(
        *(modify_resp_with_limit(rk, rt) for rk, rt in items),
        return_exceptions=True,
    )

    modified_responsibilities: Dict[str, ResponsibilityMatch] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.error("Task failed with exception: %s", r)
            continue
        if not (isinstance(r, tuple) and len(r) == 2):
            logger.warning("Unexpected result format: %r", r)
            continue

        resp_key, modifications = r
        # one result per resp_key in filtered mode; overwrite is fine
        modified_responsibilities[str(resp_key)] = modifications

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Built modified_responsibilities | resps=%d",
            len(modified_responsibilities),
        )

    try:
        validated = ResponsibilityMatches(responsibilities=modified_responsibilities)
        await asyncio.sleep(0.1)
        return validated
    except ValidationError as e:
        logger.error("Validation error when creating ResponsibilityMatches: %s", e)
        raise ValueError("Failed to validate modified responsibilities.") from e


# async def modify_multi_resps_based_on_reqs_async(
#     responsibilities: Dict[str, str],
#     requirements: Dict[str, str],
#     llm_provider: str,
#     model_id: str,
#     no_of_concurrent_workers: int = 5,
#     eligible_map: Optional[Dict[str, Set[str]]] = None,
# ) -> ResponsibilityMatches:
#     """
#     * Async version of the modify_multi_resps_based_on_reqs function.

#     Modify multiple responsibilities by aligning them with multiple job requirements.

#     This function processes multiple responsibilities by aligning each responsibility
#     with multiple job requirements. It uses the `TextEditor` class to perform the
#     modifications and executes the processing in parallel using 'joblib' to speed up
#     the process, especially when dealing with large datasets.

#     Each responsibility undergoes a three-step modification process:
#     1. Semantic Alignment: Ensures that the responsibility text matches the meaning
#     of the job requirement.
#     2. Entailment Alignment: Ensures that the responsibility text can be logically
#     inferred from the job requirement.
#     3. Dependency Parsing Alignment (DP): Ensures that the structure of the
#     responsibility text is preserved while aligning it with the job requirement.

#     Args:
#         responsibilities (dict): A dictionary of responsibility texts, where keys are
#             unique identifiers and values are the responsibility texts.
#         requirements (dict): A dictionary of job requirement texts, where keys
#             are unique requirement identifiers and values are the requirement texts.
#         llm_provider (str, optional): The name of the model to be used (e.g., "openai").
#         model_id (str, optional): The specific model version to be used
#             (e.g., "gpt-3.5-turbo").
#             Defaults to "gpt-4.1-nano".
#         n_jobs (int, optional): The number of parallel jobs to run. Defaults to -1,
#             which means using all available processors.

#     Returns:
#         * ResponsibilityMatches:
#             Pydantic object of a dictionary where keys are responsibility identifiers
#             and values are dictionaries of modified responsibility texts, each aligned
#             with multiple job requirements.

#     Example:
#         >>> modify_multi_resps_based_on_reqs(
#                 responsibilities={"resp1": "Managed a team of 5 developers"},
#                 requirements={"req1": "Experience leading software development teams."},
#                 TextEditor=TextEditor,
#                 model="openai",
#                 model_id="gpt-4.1-nano",
#                 n_jobs=-1
#             )
#     """

#     # Limit the number of concurrent tasks (in this case, coroutines) that
#     # can run simultaneously
#     semaphore = asyncio.Semaphore(no_of_concurrent_workers)  # Adjust limit as needed

#     async def modify_resp_with_limit(resp_key: str, resp: str):

#         # Decide which requirements this responsibility is allowed to use
#         if eligible_map is None:
#             reqs_sub = requirements
#         else:
#             keep = eligible_map.get(resp_key, set())
#             if not keep:
#                 # No eligible pairs → no LLM call
#                 logger.info(f"⏭️ SKIP {resp_key} (no eligible requirements)")
#                 return (resp_key, ResponsibilityMatch(optimized_by_requirements={}))
#             reqs_sub = {k: requirements[k] for k in keep if k in requirements}
#             if not reqs_sub:
#                 logger.info(
#                     f"⏭️ SKIP {resp_key} (eligible keys missing from requirements)"
#                 )
#                 return (resp_key, ResponsibilityMatch(optimized_by_requirements={}))

#         async with semaphore:
#             logger.info(
#                 f"🔄 START processing {resp_key} | reqs={len(reqs_sub)}"
#             )  # Log before starting
#             result = await modify_resp_based_on_reqs_async(
#                 resp_key, resp, reqs_sub, llm_provider, model_id
#             )
#             logger.info(
#                 f"✅ DONE processing {resp_key} | optimized={len(result[1].optimized_by_requirements)}"
#             )
#             return result

#     items = (
#         [
#             (rk, responsibilities[rk])
#             for rk in eligible_map.keys()
#             if rk in responsibilities
#         ]
#         if eligible_map is not None
#         else list(responsibilities.items())
#     )

#     tasks = [modify_resp_with_limit(rk, rt) for rk, rt in items]

#     results = await asyncio.gather(*tasks, return_exceptions=True)

#     # Handle exceptions in results if any
#     modified_responsibilities = {}
#     for result in results:
#         if isinstance(result, Exception):
#             logger.error(f"Task failed with exception: {result}")
#         elif (
#             isinstance(result, tuple) and len(result) == 2
#         ):  # Check if result is a tuple with 2 parameters (key and text)
#             # Unpack only if it's a tuple with expected length
#             resp_key, modifications = result

#             # Get or initialize ResponsibilityMatch (setdefault fetches existing
#             # data first and then add new)
#             resp_match = modified_responsibilities.setdefault(
#                 resp_key, ResponsibilityMatch(optimized_by_requirements={})
#             )

#             # Update existing requirement matches
#             resp_match.optimized_by_requirements.update(
#                 modifications.optimized_by_requirements
#             )

#         else:
#             logger.warning(f"Unexpected result format: {result}")

#     logger.info(
#         f"Before validated by ResponsibilityMatches: \n{modified_responsibilities}"
#     )  # TODO: for debugging; delete afterwards

#     # Validate and wrap the final result in ResponsibilityMatches model
#     try:
#         validated_modified_responsibilities = ResponsibilityMatches(
#             responsibilities=modified_responsibilities
#         )

#         logger.info(
#             f"After validated by ResponsibilityMatches: \n{validated_modified_responsibilities}"
#         )  # TODO: for debugging; delete afterwards

#         # Ensure event loop does not stall due to rapid execution (rate limit issue)
#         await asyncio.sleep(0.1)

#         return validated_modified_responsibilities

#     except ValidationError as e:
#         logger.error(f"Validation error when creating ResponsibilityMatches: {e}")
#         raise ValueError("Failed to validate modified responsibilities.")
