"""
pipelines_with_fsm/job_postings_manual_import_pipeline_async_fsm.py

* JSON → Job Description (DB) fallback

This mini pipeline provides a **manual override / fallback** path for the
Job Postings stage:

- Use when web scraping / LLM parsing failed for a URL (status=ERROR).
- You paste the job description into an LLM (ChatGPT web, etc.),
  get back a `JobSiteResponse`-shaped JSON, and add it into your existing
  `job_postings.json` file keyed by URL.
- This pipeline then:
    - Reads that JSON file
    - Pulls ERROR rows from `pipeline_control` at stage=JOB_POSTINGS
    - For each (url, iteration):
        * Looks up the URL in the JSON file
        * Validates + flattens
        * Inserts into `job_postings` DuckDB table
        * Marks the row COMPLETED and steps the FSM

It does **not** attempt to scrape or call LLMs. It only trusts the JSON file.

Prompt to generate the job description manually in LLM (web version):

------------------------------------------------------------------------------------

You are helping me manually generate a standardized JSON entry for a job posting.

Please output only JSON, no explanations, following this schema exactly:

{
  "<URL>": {
    "url": "<URL>",
    "company": "<Company Name>",
    "job_title": "<Exact Title>",
    "location": "<City, State or Remote or blank>",
    "content": {
        "description": "<Full cleaned job description text, no HTML>",
        "responsibilities": "<Bullet list of responsibilities as a single clean string>",
        "requirements": "<Bullet list of requirements as a single clean string>",
        "qualifications": "<Optional. Leave empty string if unknown>",
        "experience": "<Optional. Leave empty string if unknown>"
      }
    },
    "status": "success",
    "message": "Manually generated fallback job posting"
  }
}


Rules:

Use the URL I provide as the JSON key and the "url" field.

Clean all text (no HTML tags, no script fragments, no navigation text).

Put responsibilities and requirements as plain, readable sentences or bullet points, but \
all stored in a single string field.

If any field is missing from the job source, use an empty string.

Do NOT invent details not present in the job posting.

Output only JSON, no Markdown, no commentary.

Now, I will paste the raw job posting text below.

Extract and convert it into the JSON format described above.
------------------------------------------------------------------------------------

"""

# Standard libraries
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Project-level imports
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.flatten_and_rehydrate import flatten_model_to_df
from job_bot.db_io.pipeline_enums import PipelineStage, PipelineStatus, TableName
from job_bot.db_io.state_sync import load_pipeline_state
from job_bot.db_io.db_utils import (
    get_claimable_worklist,
    try_claim_one,
    release_one,
    generate_worker_id,
)

from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

from job_bot.models.resume_job_description_io_models import (
    JobPostingsBatch,
    JobSiteResponse,
)
from job_bot.utils.pydantic_model_loaders_for_files import (
    load_job_postings_file_model,
)
from job_bot.config.project_config import (
    OPENAI,
    GPT_4_1_NANO,
    JOB_POSTINGS_JSON_FILE,
)

logger = logging.getLogger(__name__)


async def persist_job_posting_from_json_async(
    url: str,
    iteration: int,
    job_postings_batch: JobPostingsBatch,
    *,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
) -> Optional[JobSiteResponse]:
    """
    Fallback worker: read a job posting from an existing JobPostingsBatch
    (loaded from job_postings.json), validate, flatten, and persist it.

    Args:
        url:
            Job posting URL (key into job_postings_batch.root).
        iteration:
            Current iteration index (part of PK in pipeline_control).
        job_postings_batch:
            In-memory batch mapping url -> JobSiteResponse.
        llm_provider:
            Only used as metadata for flatten_model_to_df; no LLM call.
        model_id:
            Only used as metadata for flatten_model_to_df; no LLM call.

    Returns:
        JobSiteResponse if persisted successfully, else None.
    """
    logger.info("📄 Manual JSON import for: %s [iter=%s]", url, iteration)

    # 1) Lookup in batch
    root_map = getattr(job_postings_batch, "root", {})
    jobposting_model = root_map.get(url)

    if jobposting_model is None:
        logger.warning("🚫 No JSON entry found for URL in job_postings.json: %s", url)
        return None

    # 2) Validate content richness (same pattern as scraping pipeline)
    data = getattr(jobposting_model, "data", None)
    content = getattr(data, "content", None)
    has_text = bool(
        isinstance(content, dict)
        and any(isinstance(v, str) and v.strip() for v in content.values())
    )

    if not has_text:
        jobposting_model.status = "error"
        jobposting_model.message = "Empty/uninformative content in manual JSON import"
        logger.warning("🚫 Manual JSON content empty/uninformative: %s", url)
    else:
        jobposting_model.status = "success"
        jobposting_model.message = "Job description imported from manual JSON fallback"

    # 3) Flatten to DF and persist
    try:
        job_posting_batch = JobPostingsBatch(root={url: jobposting_model})  # type: ignore[arg-type]
        job_df = flatten_model_to_df(
            model=job_posting_batch,
            table_name=TableName.JOB_POSTINGS,
            iteration=iteration,
            llm_provider=llm_provider,
            model_id=model_id,
        )

        # insert_df_with_config is sync; run in thread pool
        await asyncio.to_thread(
            insert_df_with_config,
            job_df,
            TableName.JOB_POSTINGS,
            url=url,
            iteration=iteration,
        )

        logger.info(
            "✅ Persisted job posting (manual JSON) for %s [iter=%s] — status: %s",
            url,
            iteration,
            jobposting_model.status,
        )
        return jobposting_model

    except Exception:
        logger.exception("💾 Manual JSON post-process or persist failed: %s", url)
        return None


async def process_job_postings_from_json_batch_async_fsm(
    items: List[Tuple[str, int]],
    *,
    job_postings_json_file: Union[str, Path],
    worker_id: str,
    lease_minutes: int = 10,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    no_of_concurrent_workers: int = 5,
) -> List[asyncio.Task]:
    """
    Prepare asyncio.Tasks for **manual JSON import** of job postings using the
    same lease + FSM model as the main job_postings pipeline.

    Each `item` is a (url, iteration) pair that is currently in ERROR at
    stage=JOB_POSTINGS.

    Flow per item:
      1) Ensure there is a pipeline_control row for the URL.
      2) Ensure FSM state is JOB_POSTINGS (stage gate).
      3) Try to claim a machine lease via try_claim_one().
      4) Run persist_job_posting_from_json_async():
            - lookup URL in job_postings.json
            - validate + flatten + insert into DuckDB
      5) Release lease with final_status = COMPLETED or ERROR.
      6) On success, call fsm.step() to advance to next stage.

    Args:
        items:
            List of (url, iteration) pairs to process.
        job_postings_json_file:
            Path to the job_postings JSON file (same one used by your file ETL).
        worker_id:
            Stable ID for this runner (generate once per process).
        lease_minutes:
            Lease duration (minutes) per claim.
        llm_provider, model_id:
            Metadata only; no LLM calls are made here.
        no_of_concurrent_workers:
            Concurrency cap via a semaphore for DB writes.

    Returns:
        List[asyncio.Task] — schedule with asyncio.gather().
    """
    file_path = Path(job_postings_json_file)
    if not file_path.exists():
        raise FileNotFoundError(f"job_postings_json_file not found: {file_path}")

    job_postings_batch = load_job_postings_file_model(file_path)
    if job_postings_batch is None:
        logger.error("❌ Failed to load/validate job postings batch from %s", file_path)
        return []  # or `return` / skip, depending on your function

    semaphore = asyncio.Semaphore(no_of_concurrent_workers)

    async def run_one(url: str, iteration: int):
        # 1) Ensure we have a pipeline state
        state = load_pipeline_state(url)
        if state is None:
            logger.warning("⚠️ No pipeline state for URL (manual import): %s", url)
            return

        # 2) Stage gate
        fsm_manager = PipelineFSMManager()
        fsm = fsm_manager.get_fsm(url)
        if fsm.state != PipelineStage.JOB_POSTINGS.value:
            logger.info(
                "⏩ Skip manual import (stage=%s, expected=JOB_POSTINGS): %s",
                fsm.state,
                url,
            )
            return

        # 3) Claim lease
        claimed = await asyncio.to_thread(
            try_claim_one,
            url=url,
            iteration=iteration,
            worker_id=worker_id,
            lease_minutes=lease_minutes,
        )
        if not claimed:
            logger.info(
                "🔒 Busy; could not claim for manual import: %s [%s]", url, iteration
            )
            return

        released = False
        try:
            # 4) Do the work under concurrency control
            async with semaphore:
                result = await persist_job_posting_from_json_async(
                    url=url,
                    iteration=iteration,
                    job_postings_batch=job_postings_batch,
                    llm_provider=llm_provider,
                    model_id=model_id,
                )

            # 5) Finalize lease + FSM
            if result is not None:
                await asyncio.to_thread(
                    release_one,
                    url=url,
                    iteration=iteration,
                    worker_id=worker_id,
                    final_status=PipelineStatus.COMPLETED,
                )
                released = True
                try:
                    fsm.step()
                except Exception as e:
                    logger.exception(
                        "FSM step() failed after manual import for %s [%s]: %s",
                        url,
                        iteration,
                        e,
                    )
            else:
                await asyncio.to_thread(
                    release_one,
                    url=url,
                    iteration=iteration,
                    worker_id=worker_id,
                    final_status=PipelineStatus.ERROR,
                )
                released = True

        except Exception as e:
            logger.exception(
                "Unhandled exception during manual import for %s [%s]: %s",
                url,
                iteration,
                e,
            )
            try:
                await asyncio.to_thread(
                    release_one,
                    url=url,
                    iteration=iteration,
                    worker_id=worker_id,
                    final_status=PipelineStatus.ERROR,
                )
                released = True
            except Exception:
                logger.exception(
                    "Failed to release lease after exception (manual import) for %s [%s]",
                    url,
                    iteration,
                )
        finally:
            if not released:
                # Extremely defensive: do not leave leases dangling
                try:
                    await asyncio.to_thread(
                        release_one,
                        url=url,
                        iteration=iteration,
                        worker_id=worker_id,
                        final_status=PipelineStatus.ERROR,
                    )
                except Exception:
                    logger.exception(
                        "Final lease cleanup failed (manual import) for %s [%s]",
                        url,
                        iteration,
                    )

    # Schedule one task per (url, iteration)
    return [asyncio.create_task(run_one(u, it)) for (u, it) in items]


async def run_job_postings_manual_import_async_fsm(
    *,
    job_postings_json_file: Union[str, Path] = JOB_POSTINGS_JSON_FILE,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_concurrent_tasks: int = 5,
    filter_keys: Optional[list[str]] = None,
    lease_minutes: int = 10,
) -> None:
    """
    Top-level runner for the **manual JSON import fallback** at the Job Postings
    stage.

    It:
      - Pulls (url, iteration) from `pipeline_control` where:
          * stage = 'job_postings'
          * status = 'ERROR'
          * task_state = 'READY'
          * lease is free/expired
      - Optionally filters by `filter_keys`.
      - For each, runs manual JSON import from `job_postings_json_file`.

    Typical usage:
        await run_job_postings_manual_import_async_fsm(
            job_postings_json_file="data/input/job_postings.json",
        )

    Args:
        job_postings_json_file:
            Path to your existing job_postings JSON file (same as file ETL).
        llm_provider, model_id:
            Metadata only; no LLM calls are made here.
        max_concurrent_tasks:
            Maximum number of parallel manual-import workers.
        filter_keys:
            Optional subset of URLs to process.
        lease_minutes:
            Lease duration for each claimed row.
    """
    # 1) Select ERROR rows at JOB_POSTINGS stage
    statuses = (PipelineStatus.ERROR,)

    worklist: list[tuple[str, int]] = get_claimable_worklist(
        stage=PipelineStage.JOB_POSTINGS,
        status=statuses,
        max_rows=max_concurrent_tasks * 4 or 1000,
    )

    # 2) Optional subset filter
    if filter_keys:
        filter_set = set(filter_keys)
        worklist = [(u, it) for (u, it) in worklist if u in filter_set]

    if not worklist:
        logger.info(
            "📭 No claimable ERROR (url, iteration) at stage 'job_postings' for manual import."
        )
        return

    # 3) Worker identity
    worker_id = generate_worker_id(prefix="job_postings_manual_import")

    logger.info(
        "🚀 Starting job_postings_manual_import pipeline | %d items | worker_id=%s",
        len(worklist),
        worker_id,
    )

    # 4) Launch batch
    tasks = await process_job_postings_from_json_batch_async_fsm(
        items=worklist,
        job_postings_json_file=job_postings_json_file,
        worker_id=worker_id,
        lease_minutes=lease_minutes,
        llm_provider=llm_provider,
        model_id=model_id,
        no_of_concurrent_workers=max_concurrent_tasks,
    )

    await asyncio.gather(*tasks)
    logger.info("✅ Finished job_postings_manual_import FSM pipeline.")
