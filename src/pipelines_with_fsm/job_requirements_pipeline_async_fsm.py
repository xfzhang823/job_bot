"""
extracted_job_requirements_pipeline_async_fsm.py

FSM-aware pipeline to extract structured job requirements from job_postings using LLMs.

Transitions:
    PipelineStage.JOB_POSTINGS → PipelineStage.EXTRACTED_REQUIREMENTS
"""

# Standard libraries
import asyncio
import logging
from typing import List, Optional
import json
from pydantic import ValidationError

# Project-level imports
from db_io.db_insert import insert_df_dedup
from db_io.db_transform import flatten_model_to_df
from db_io.pipeline_enums import PipelineStage, PipelineStatus, TableName
from db_io.state_sync import load_pipeline_state
from db_io.db_utils import get_urls_by_stage_and_status
from fsm.pipeline_fsm_manager import PipelineFSMManager
from utils.pydantic_model_loaders_from_db import load_job_postings_for_url_from_db

from models.llm_response_models import JobSiteResponse
from models.resume_job_description_io_models import (
    ExtractedRequirementsBatch,
    RequirementsResponse,
)
from project_config import OPENAI, GPT_4_TURBO, ANTHROPIC, CLAUDE_HAIKU
from preprocessing.extract_requirements_with_llms_async import (
    extract_job_requirements_with_openai_async,
    extract_job_requirements_with_anthropic_async,
)

logger = logging.getLogger(__name__)


async def extract_persist_requirements_for_url(
    url: str,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
    model_id: str = GPT_4_TURBO,
    llm_provider: str = OPENAI,
) -> Optional[RequirementsResponse]:
    logger.info(f"🧠 Starting requirements extraction for: {url}")
    fsm = fsm_manager.get_fsm(url)

    # Load from DuckDB
    job_description_model = load_job_postings_for_url_from_db(
        url=url,
        status=None,
        iteration=0,
    )
    job_posting: JobSiteResponse = job_description_model.root[url]  # type: ignore[attr-defined]
    data_section = job_posting.data.model_dump()
    content = data_section.get("content")

    # Validate job content
    if not isinstance(content, dict) or not any(
        v.strip() for v in content.values() if isinstance(v, str)
    ):
        logger.warning(f"🚫 Skipping {url} — empty or invalid job content.")
        fsm.mark_status(PipelineStatus.ERROR, notes="Empty or invalid job content.")
        return None

    job_description_json = json.dumps(data_section, indent=2)

    # Step 3: Extract via LLM
    try:
        async with semaphore:
            if llm_provider.lower() == OPENAI:
                requirements_model = await extract_job_requirements_with_openai_async(
                    job_description=job_description_json, model_id=model_id
                )
            elif llm_provider.lower() == ANTHROPIC:
                requirements_model = (
                    await extract_job_requirements_with_anthropic_async(
                        job_description=job_description_json, model_id=model_id
                    )
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    except Exception as e:
        logger.exception(f"❌ LLM extraction failed for {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="LLM extraction failed")
        return None

    # ✅ Step 4: Persist to DuckDB
    try:
        df = flatten_model_to_df(
            model=ExtractedRequirementsBatch(root={url: requirements_model}),  # type: ignore[attr-defined]
            table_name=TableName.EXTRACTED_REQUIREMENTS,
            stage=PipelineStage.EXTRACTED_REQUIREMENTS,
        )
        insert_df_dedup(df, TableName.EXTRACTED_REQUIREMENTS.value)

        # ✅ FSM Step + Status
        fsm.step()
        fsm.mark_status(
            PipelineStatus.IN_PROGRESS,
            notes="Requirements extracted and saved to DB.",
        )

        logger.info(f"✅ Completed extraction for {url}")
        return requirements_model

    except Exception as e:
        logger.exception(f"❌ Failed to insert or step FSM for {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="Post-processing failed")
        return None


async def process_extracted_requirements_batch_async_fsm(
    urls: List[str],
    llm_provider: str,
    model_id: str,
    no_of_concurrent_workers: int = 5,
) -> List[asyncio.Task]:
    semaphore = asyncio.Semaphore(no_of_concurrent_workers)

    async def run_one(url: str):
        state = load_pipeline_state(url)
        if state is None:
            logger.warning(f"⚠️ No pipeline state for URL: {url}")
            return

        fsm_manager = PipelineFSMManager()
        fsm = fsm_manager.get_fsm(url)
        if fsm.get_current_stage() != PipelineStage.JOB_POSTINGS.value:
            logger.info(f"⏩ Skipping {url}, current stage: {fsm.state}")
            return

        await extract_persist_requirements_for_url(
            url=url,
            fsm_manager=fsm_manager,
            semaphore=semaphore,
            llm_provider=llm_provider,
            model_id=model_id,
        )

    return [asyncio.create_task(run_one(url)) for url in urls]


async def run_job_requirements_pipeline_async_fsm(
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_TURBO,
    max_concurrent_tasks: int = 5,
    filter_keys: Optional[list[str]] = None,
):
    urls = get_urls_by_stage_and_status(
        stage=PipelineStage.JOB_POSTINGS,
        status=PipelineStatus.NEW,
    )

    if filter_keys:
        urls = [u for u in urls if u in filter_keys]

    if not urls:
        logger.info("📭 No job URLs to process at 'job_postings' stage.")
        return

    logger.info(f"🧪 Extracting requirements for {len(urls)} URLs...")

    tasks = await process_extracted_requirements_batch_async_fsm(
        urls=urls,
        llm_provider=llm_provider,
        model_id=model_id,
        no_of_concurrent_workers=max_concurrent_tasks,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished job requirements FSM pipeline.")
