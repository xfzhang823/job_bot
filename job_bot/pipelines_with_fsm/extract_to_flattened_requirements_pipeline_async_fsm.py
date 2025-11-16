"""
pipelines_with_fsm/extract_job_requirements_pipeline_async_fsm.py

* Job Description (DB) → Requirements (DB)

FSM-aware, DuckDB-native pipeline to extract structured job requirements from
job_postings using LLMs. No filesystem dependency.

Stage transition intent:
    JOB_POSTINGS  --(extract)-->  FLATTENED_REQUIREMENTS

Control-table semantics per URL:
- When starting this stage from JOB_POSTINGS, mark that stage IN_PROGRESS.
- On success: mark JOB_POSTINGS COMPLETE, step() to FLATTED_REQUIREMENTS,
  then mark NEW for the new stage (so the next stage runner can pick it up).
- On failure: mark ERROR and do not advance.
"""

from __future__ import annotations

# Standard libraries
import asyncio
import json
import logging
from typing import Optional

# Project-level imports (keep paths consistent with your repo layout)
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.flatten_and_rehydrate import flatten_model_to_df
from job_bot.db_io.pipeline_enums import PipelineStage, PipelineStatus, TableName
from job_bot.db_io.state_sync import load_pipeline_state

# from job_bot.db_io.db_utils import get_urls_ready_for_transition
from job_bot.db_io.db_utils import (
    get_claimable_worklist,
    try_claim_one,
    generate_worker_id,
    finalize_one_row_in_pipeline_control,
)

from job_bot.db_io.db_loaders import load_table

from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

from job_bot.models.llm_response_models import JobSiteResponse  # data: JobSiteData
from job_bot.models.resume_job_description_io_models import (
    # NestedRequirementsBatch,
    RequirementsResponse,
    Requirements,
)
from job_bot.preprocessing.extract_requirements_with_llms_async import (
    extract_job_requirements_with_openai_async,
    extract_job_requirements_with_anthropic_async,
)

from job_bot.config.project_config import OPENAI, GPT_4_1_NANO, ANTHROPIC, CLAUDE_HAIKU


logger = logging.getLogger(__name__)


async def extract_and_persist_requirements_for_url(
    url: str,
    *,
    iteration: int,
    semaphore: asyncio.Semaphore,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
) -> Optional[RequirementsResponse]:
    """
    Orchestrate requirements extraction → flattened insert for a single job URL
    (FSM + DuckDB, no filesystem).

    This function:
    • Marks the current control row `IN_PROGRESS` (stage = FLATTENED_REQUIREMENTS).
    • Loads the structured job posting for `url` from DuckDB (`job_postings`).
    • Validates/serializes the posting into a JSON payload for the LLM.
    • Uses a bounded semaphore to call the LLM (via `_extract_job_requirements`)
        and returns a typed `RequirementsResponse`.
    • Flattens the response and **projects to the flattened schema**
        (`url`, `requirement_key`, `requirement`, `source_file`, plus LLM provenance),
        then inserts into DuckDB **`flattened_requirements`** with de-duplication.
        (Note: `iteration` is inherited from `pipeline_control` via inserter config.)

    Side effects:
    - Writes to DuckDB table: **`flattened_requirements`**.
    - Updates `pipeline_control` status and performs a stage transition.
    - Makes a network call to the selected LLM provider.

    Concurrency:
    - LLM calls are rate-limited by `semaphore`.
    - Safe to run across many URLs concurrently when tasks share a process-wide
        `PipelineFSMManager`.

    Idempotency & behavior:
    - If the URL is not currently at `FLATTENED_REQUIREMENTS`, the runner should
        skip/no-op (worklist selection enforces this upstream).
    - On any error (load/validate/LLM/DB/FSM), the function logs, marks `ERROR`,
        **does not advance** the FSM, and returns `None`.
    - On success, returns the `RequirementsResponse`.

    Args:
    url: Canonical job URL key.
    fsm_manager: Manager used to fetch and update the URL's FSM state.
    semaphore: Concurrency limiter for the LLM call.
    model_id: LLM model identifier (e.g., `GPT_4_1_NANO`).
    llm_provider: LLM provider name (e.g., `OPENAI`, `ANTHROPIC`).

    Returns:
    RequirementsResponse on success; otherwise `None`.
    """

    logger.info(f"🧠 Starting requirements extraction for: {url}")

    try:
        # Blocking DB read → thread
        postings_batch = await asyncio.to_thread(
            load_table, TableName.JOB_POSTINGS, url=url, iteration=iteration
        )
        job_posting: JobSiteResponse = postings_batch.root[url]

        job_description_json = _validate_job_content(job_posting)
        if not job_description_json:
            logger.warning(f"🚫 Skipping {url} — empty/invalid job content.")
            return None

        # LLM (bounded)
        async with semaphore:
            requirements_response_model = await _extract_job_requirements(
                job_description_json=job_description_json,
                llm_provider=llm_provider,
                model_id=model_id,
            )

        # Step 1: normalized nested dict
        nested_requirements = requirements_response_model.ensure_requirements_dict()
        # nested: Dict[str, List[str]]

        # flatten -> Requirements
        flat_req: dict[str, str] = {}
        for cat_idx, (category, items) in enumerate(
            sorted(nested_requirements.items())
        ):
            cat_norm = str(category).strip().lower().replace(" ", "_")
            for item_idx, text in enumerate(items, start=1):
                key = f"{cat_idx}.{cat_norm}.{item_idx}"
                flat_req[key] = text.strip()

        requirements_model = Requirements(url=url, requirements=flat_req)

        # Flatten → insert (blocking DB write → thread)
        df = flatten_model_to_df(
            model=requirements_model,
            table_name=TableName.FLATTENED_REQUIREMENTS,
            llm_provider=llm_provider,
            model_id=model_id,
        )
        if df is None:
            logger.error("Flatten failed, got None instead of DataFrame")
            return None

        await asyncio.to_thread(
            insert_df_with_config,
            df,
            TableName.FLATTENED_REQUIREMENTS,
            url=url,  # inserter stamps iteration per YAML
            llm_provider=llm_provider,
            model_id=model_id,
            iteration=iteration,
        )

        logger.info(f"✅ Completed extraction for {url}")
        return requirements_response_model

    except Exception:
        logger.exception(f"❌ extract_and_persist failed for {url}")
        return None


async def process_extract_and_flatten_requirements_batch_async_fsm(
    url_iter_pairs: list[tuple[str, int]],
    *,
    worker_id: str,
    llm_provider: str,
    model_id: str,
    no_of_concurrent_workers: int = 5,
) -> list[asyncio.Task]:
    """Claim → run pure op → finalize/step. Mirrors job_postings pattern."""
    semaphore = asyncio.Semaphore(no_of_concurrent_workers)
    fsm_manager = PipelineFSMManager()

    async def run_one(url: str, iteration: int) -> None:
        # Pre-flight check (optional)
        if load_pipeline_state(url) is None:
            logger.warning(f"⚠️ No pipeline_state row for {url}")

        # Lease: claim or skip
        if not try_claim_one(url=url, iteration=iteration, worker_id=worker_id):
            logger.info(f"⏭️ Skipping {url}@{iteration} — already claimed.")
            return

        try:
            result = await extract_and_persist_requirements_for_url(
                url,
                iteration=iteration,
                semaphore=semaphore,
                llm_provider=llm_provider,
                model_id=model_id,
            )

            ok = result is not None
            finalized = finalize_one_row_in_pipeline_control(
                url=url,
                iteration=iteration,
                worker_id=worker_id,
                ok=ok,
                notes="Requirements saved to DB" if ok else "Operation returned None",
            )
            if not finalized:
                logger.warning(
                    f"[finalize] Lost lease for {url}@{iteration}; not stepping."
                )
                return

            # Advance FSM only on success
            if ok:
                fsm_manager.get_fsm(url).step()

        except Exception as e:
            logger.exception(f"❌ Failure in run_one for {url}@{iteration}: {e}")
            # Best-effort error finalize (still lease-validated)
            finalized = finalize_one_row_in_pipeline_control(
                url=url,
                iteration=iteration,
                worker_id=worker_id,
                ok=False,
                notes=f"extract/flatten failed: {e}",
            )
            if not finalized:
                logger.warning(
                    f"[finalize] Could not mark ERROR for {url}@{iteration} (lease mismatch)."
                )

    return [asyncio.create_task(run_one(u, it)) for (u, it) in url_iter_pairs]


async def run_extract_to_flattened_requirements_pipeline_async_fsm(
    *,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_concurrent_tasks: int = 5,
    retry_errors: bool = False,
    filter_keys: Optional[list[str]] = None,
) -> None:
    """Entry point: process all URLs at JOB_POSTINGS stage with status NEW.

    filter_keys: if provided, limit to this subset of URLs.
    """
    stage_enum = PipelineStage.FLATTENED_REQUIREMENTS

    statuses = (
        (PipelineStatus.NEW, PipelineStatus.ERROR, PipelineStatus.IN_PROGRESS)
        if retry_errors
        else (PipelineStatus.NEW,)
    )
    # Expecting list[tuple[str, int]]; if your helper returns dicts, coerce here.
    work = get_claimable_worklist(
        stage=stage_enum, status=statuses
    )  # e.g., [("https://...", 0), ...]
    # work: list[tuple[str, int]] = work  # or: [(w["url"], int(w["iteration"])) for w in work_raw]

    worker_id = generate_worker_id("reqs")
    logger.info(f"worklist size: {len(work)} (stage={stage_enum})")

    # Filter by URLs (Optional)
    if filter_keys:
        url_iter_pairs = [(u, it) for (u, it) in work if u in filter_keys]
    else:
        url_iter_pairs = work

    if not url_iter_pairs:
        logger.info("📭 No claimable rows at FLATTENED_REQUIREMENTS.")
        return

    logger.info(f"🧪 Extracting requirements for {len(url_iter_pairs)} URL(s)…")

    tasks = await process_extract_and_flatten_requirements_batch_async_fsm(
        url_iter_pairs,
        worker_id=worker_id,
        llm_provider=llm_provider,
        model_id=model_id,
        no_of_concurrent_workers=max_concurrent_tasks,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished job requirements FSM pipeline.")


async def _extract_job_requirements(
    *, job_description_json: str, llm_provider: str, model_id: str
) -> RequirementsResponse:
    """Dispatch to provider-specific async extraction."""
    provider = llm_provider.lower()
    if provider == OPENAI:
        return await extract_job_requirements_with_openai_async(
            job_description=job_description_json, model_id=model_id
        )
    if provider == ANTHROPIC:
        return await extract_job_requirements_with_anthropic_async(
            job_description=job_description_json, model_id=model_id
        )
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def _validate_job_content(job_posting: JobSiteResponse) -> Optional[str]:
    """
    Return pretty JSON payload if content exists and is non-empty; else None.

    NOTE:
      - `job_posting.data` is a Pydantic model (JobSiteData), not a dict.
      - We serialize the whole `data` section so the LLM sees title/company/etc.
    """
    data_model = job_posting.data  # JobSiteData
    content = data_model.content

    if not isinstance(content, dict):
        return None

    # Non-empty if any string field has non-blank content
    any_text = any((v.strip() for v in content.values() if isinstance(v, str)))
    if not any_text:
        return None

    # Pretty JSON string for LLM input
    return json.dumps(data_model.model_dump(), indent=2)
