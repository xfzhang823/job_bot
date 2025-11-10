"""
job_postings_pipeline_async_fsm.py

* Web → Job Description (DB)

This module implements the asynchronous FSM-aware pipeline for scraping,
parsing, validating, and persisting job posting webpages into job description
content in DuckDB table.

This pipeline:
- Pulls NEW URLs at stage=JOB_URLS
- Scrapes + parses each URL (LLM-assisted)
- Validates + persists to DuckDB (job_postings)
- Steps FSM → JOB_POSTINGS (IN_PROGRESS on success)

Pipeline Stage:
    PipelineStage.JOB_URLS → PipelineStage.JOB_POSTINGS

FSM Controls:
    - Each job (URL) has an associated PipelineState
    - PipelineFSM manages allowed transitions and persists progress
    - FSM steps forward only after successful job scrape + persist

Concurrency:
    - Semaphore used to limit the number of concurrent scraping + LLM operations
    - Ensures safe and resource-efficient processing across batches

Dependencies:
    - process_webpages_to_json_async(): Core scraping + parsing logic
    - flatten_model_to_df(): Converts parsed model to DuckDB-ready DataFrame
    - insert_df_with_config(): Inserts deduplicated records into DuckDB
    - PipelineFSMManager: Controls per-URL state and stage progression

---
📦 Example Usage (in orchestrator):

    await run_job_postings_pipeline_async_fsm()

---
🔁 ASCII Flowchart:

+---------------------------+
| pipeline_control table    |
| (stage = 'job_urls')      |
+------------+--------------+
             |
             v
+--------------------------------------+
|  process_job_posting_batch_async_fsm |
|  - uses FSM                          |
|  - limits concurrency                |
+------------+-------------------------+
             |
             v
+----------------------------------------+
| scrape_parse_persist_job_posting_async |
| - scrape + LLM parse                   |
| - validate content                     |
| - flatten + insert to DB               |
| - fsm.step() if success                |
+------------+---------------------------+
             |
             v
+----------------------------+
| stage = 'job_postings'     |
| status = 'in_progress'     |
| persisted in DuckDB        |
+----------------------------+

---
"""

# Standard libraries
import asyncio
import logging
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


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


# --- IO surface: function-style db_loaders ---
# from job_bot.db_io.db_loaders import (
#     persist_job_postings_batch,
# )
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

from job_bot.models.resume_job_description_io_models import (
    JobPostingsBatch,
    JobSiteResponse,
)
from job_bot.config.project_config import (
    OPENAI,
    GPT_4_1_NANO,
)

from job_bot.utils.webpage_reader_async import process_webpages_to_json_async


# Set logger
logger = logging.getLogger(__name__)


async def scrape_parse_persist_job_posting_async(
    url: str,
    iteration: int,
    semaphore: asyncio.Semaphore,
    *,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> Optional[JobSiteResponse]:
    """
    Scrapes, parses, validates, and persists a single job posting from the web.

    Args:
        url: Job posting URL.
        iteration: Current iteration index (part of PK in pipeline_control).
        fsm_manager: FSM manager controlling pipeline state transitions.
        semaphore: Concurrency limiter for I/O-intensive steps.
        llm_provider: LLM provider ("openai", "anthropic", etc.).
        model_id: LLM model identifier.
        max_tokens: Maximum tokens for LLM generation.
        temperature: Sampling temperature.

    Returns:
        Optional[JobSiteResponse]: Parsed job posting model, or None on failure.


    Note:
      - This function performs the work only.
      - Do NOT mark/release leases or mutate FSM state here.
      - The caller (batch orchestrator) handles claim/release and fsm.step().
    """
    logger.info("🌐 Starting scrape/parse for: %s [iter=%s]", url, iteration)

    # 1) Scrape + LLM parse (I/O under semaphore)
    try:
        async with semaphore:
            job_postings_batch_model = await process_webpages_to_json_async(
                urls=url,
                llm_provider=llm_provider,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
    except Exception:
        logger.exception("❌ Scraping or LLM failed: %s", url)
        return None

    # 2) Validate + Flatten + Persist
    try:
        # Defensive lookup for the URL key
        root_map = getattr(job_postings_batch_model, "root", {})  # type: ignore[attr-defined]
        jobposting_model = root_map.get(url)
        if jobposting_model is None:
            logger.warning("🚫 Parser returned no entry for URL key: %s", url)
            return None

        # Validate content richness
        data = getattr(jobposting_model, "data", None)
        content = getattr(data, "content", None)
        has_text = bool(
            isinstance(content, dict)
            and any(isinstance(v, str) and v.strip() for v in content.values())
        )

        if not has_text:
            jobposting_model.status = "error"
            jobposting_model.message = "Empty/uninformative content extracted"
            logger.warning("🚫 Skipping (empty content): %s", url)
        else:
            jobposting_model.status = "success"
            jobposting_model.message = "Job description parsed successfully"

        # Flatten to DF and persist (sync call off-thread)
        job_posting_batch = JobPostingsBatch(root={url: jobposting_model})  # type: ignore[arg-type]
        job_df = flatten_model_to_df(
            model=job_posting_batch,
            table_name=TableName.JOB_POSTINGS,
            iteration=iteration,  # <- use the iteration provided by control-plane
            llm_provider=llm_provider,
            model_id=model_id,
        )
        await asyncio.to_thread(
            insert_df_with_config,
            job_df,
            TableName.JOB_POSTINGS,
            url=url,
            iteration=iteration,
        )

        logger.info(
            "✅ Persisted job posting for %s [iter=%s] — status: %s",
            url,
            iteration,
            jobposting_model.status,
        )
        return jobposting_model

    except Exception:
        logger.exception("💾 Post-process or persist failed: %s", url)
        return None


async def process_job_postings_batch_async_fsm(
    items: List[Tuple[str, int]],
    *,
    worker_id: str,
    lease_minutes: int = 10,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    no_of_concurrent_workers: int = 5,
) -> List[asyncio.Task]:
    """
    Prepare asyncio.Tasks for scraping/parsing/persisting job postings using the
    lease + human-gate FSM model.

    Each `item` is a (url, iteration) pair. For each item:
      1) Verify it's at stage = JOB_URLS (FSM).
      2) Try to claim (machine lease) scoped to (url, iteration).
      3) Do the work (scrape → parse → persist).
      4) Release the lease with final_status = COMPLETED or ERROR.
      5) On success, advance FSM to next stage via fsm.step().

    Args:
        items: List of (url, iteration) pairs considered claimable upstream.
        worker_id: Stable ID for this runner (generate once per process).
        lease_seconds: Lease duration per claim (auto-expire if worker dies).
        llm_provider, model_id, max_tokens, temperature: LLM config for downstream.
        no_of_concurrent_workers: Concurrency cap via a semaphore.

    Returns:
        List[asyncio.Task] — schedule with asyncio.gather().
    """
    semaphore = asyncio.Semaphore(no_of_concurrent_workers)

    async def run_one(url: str, iteration: int):
        # Optional: ensure we have a control row; if not, skip
        state = load_pipeline_state(
            url
        )  # if your loader also needs iteration, adjust here
        if state is None:
            logger.warning("⚠️ No pipeline state for URL: %s", url)
            return

        # Stage gate (human gate is enforced upstream in the claimable query)
        fsm_manager = PipelineFSMManager()
        fsm = fsm_manager.get_fsm(url)
        if fsm.state != PipelineStage.JOB_URLS.value:
            logger.info("⏩ Skip (stage=%s): %s", fsm.state, url)
            return

        # Machine lease claim (single worker for (url, iteration))
        claimed = await asyncio.to_thread(
            try_claim_one,
            url=url,
            iteration=iteration,
            worker_id=worker_id,
            lease_minutes=lease_minutes,
        )
        if not claimed:
            logger.info("🔒 Busy; could not claim: %s [%s]", url, iteration)
            return

        released = False
        try:
            # Do the work under concurrency control
            async with semaphore:
                result = await scrape_parse_persist_job_posting_async(
                    url=url,
                    iteration=iteration,
                    semaphore=semaphore,  # you may remove if the callee doesn't need it
                    llm_provider=llm_provider,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            # Decide finalization
            if result is not None:
                # Persist succeeded; release as COMPLETED and advance FSM
                await asyncio.to_thread(
                    release_one,
                    url=url,
                    iteration=iteration,
                    worker_id=worker_id,
                    final_status=PipelineStatus.COMPLETED,
                )
                released = True
                try:
                    fsm.step()  # stage-wise progression after successful persist
                except Exception as e:
                    # If step fails, keep the row COMPLETED (data is saved);
                    # log for retry of FSM step
                    logger.exception(
                        "FSM step() failed for %s [%s]: %s", url, iteration, e
                    )
            else:
                # Worker ran but produced no result → ERROR
                await asyncio.to_thread(
                    release_one,
                    url=url,
                    iteration=iteration,
                    worker_id=worker_id,
                    final_status=PipelineStatus.ERROR,
                )
                released = True

        except Exception as e:
            logger.exception("Unhandled exception for %s [%s]: %s", url, iteration, e)
            # Ensure we release with ERROR if something blew up and we still hold the lease
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
                    "Failed to release lease after exception for %s [%s]",
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
                        "Final lease cleanup failed for %s [%s]", url, iteration
                    )

    # Schedule one task per (url, iteration)
    return [asyncio.create_task(run_one(u, it)) for (u, it) in items]


async def run_job_postings_pipeline_async_fsm(
    *,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    max_concurrent_tasks: int = 5,
    filter_keys: Optional[list[str]] = None,
    retry_errors: bool = False,
    lease_minutes: int = 10,
) -> None:
    """
    Asynchronous FSM-aware pipeline for the **Job Postings** stage.

    This stage pulls job URLs from the `pipeline_control` table at the
    `job_urls` stage, claims eligible rows using the lease + human-gate
    concurrency model, and then runs scraping and parsing tasks in parallel.

    -------------------------------------------------------------------------
    🔁  High-Level Flow
    -------------------------------------------------------------------------
    1. **Worklist selection (human-gate + lease aware)**
       - Pulls all (url, iteration) pairs where:
         - `stage = 'job_urls'`
         - `status IN ('NEW', 'ERROR' if retry_errors=True)`
         - `task_state = 'READY'` (human gate open)
         - `(is_claimed = FALSE OR lease_until < now())` (not leased)
       - These are potential rows for this machine to process.

    2. **Optional filtering**
       - If `filter_keys` is provided, only URLs in that list are processed.

    3. **Worker identity**
       - A `worker_id` is generated per process (e.g. "job_postings_20251106_abc123").
       - Used only for `try_claim_one()` / `release_one()` to enforce single-worker
         ownership of each `(url, iteration)` row during the lease period.

    4. **Parallel batch execution**
       - For each claimable `(url, iteration)`:
         a. `try_claim_one()` → sets `is_claimed=TRUE`, `worker_id`, and `lease_until`
         b. `scrape_parse_persist_job_posting_async()` → performs the actual scraping,
            parsing, model validation, and DuckDB persistence (no FSM mutation).
         c. `release_one()` → safely releases the lease with final status
            (`COMPLETED` or `ERROR`), and clears `worker_id`.
         d. `fsm.step()` → advances the FSM to the next stage if persisted successfully.

    5. **Concurrency management**
       - A shared asyncio semaphore (`max_concurrent_tasks`) limits active
         scrape/parse workers.
       - `asyncio.gather()` waits for all tasks to finish before exiting.

    6. **Iteration tracking**
       - `iteration` comes from the control table (part of primary key `(url, iteration)`).
       - It enables multiple processing passes over the same URL across different runs.
       - For now, iteration may be 0 globally — future-safe for versioned reprocessing.

    -------------------------------------------------------------------------
    ⚙️  Parameters
    -------------------------------------------------------------------------
    llm_provider (str)
        Name of the LLM provider to use (e.g., "openai", "anthropic").
    model_id (str)
        Model identifier to pass to the LLM (e.g., "gpt-4-1-nano").
    max_tokens (int)
        Maximum token limit for LLM outputs.
    temperature (float)
        Sampling temperature for LLM generation.
    max_concurrent_tasks (int)
        Maximum number of parallel async scraping workers.
    filter_keys (Optional[list[str]])
        Subset of URLs to process. If provided, all other URLs are ignored.
    retry_errors (bool)
        Whether to include rows currently marked as `ERROR` in addition to `NEW`.
    lease_minutes (int)
        Duration (in minutes) of the machine lease per claimed row. If the
        worker crashes or hangs beyond this time, other workers can reclaim it.

    -------------------------------------------------------------------------
    🧱  Side Effects
    -------------------------------------------------------------------------
    - Updates `pipeline_control` rows via:
        * `try_claim_one()` — mark as claimed, set lease time.
        * `release_one()` — mark as completed or error, clear claim.
    - Writes parsed job postings into `job_postings` DuckDB table.
    - Advances FSM stage on successful completion.

    -------------------------------------------------------------------------
    🧾  Example
    -------------------------------------------------------------------------
    ```python
    await run_job_postings_pipeline_async_fsm(
        llm_provider="openai",
        model_id="gpt-4-1-nano",
        max_concurrent_tasks=10,
        retry_errors=True,
    )
    ```

    -------------------------------------------------------------------------
    ✅  Summary
    -------------------------------------------------------------------------
    This function ensures that:
      - Only human-approved (`READY`) and unclaimed jobs are processed.
      - Each (url, iteration) is handled by exactly one worker at a time.
      - Failures don’t block other URLs from progressing.
      - FSM transitions remain consistent and idempotent.
    """
    # 1) Determine which statuses to include
    statuses = (
        (PipelineStatus.NEW, PipelineStatus.ERROR)
        if retry_errors
        else (PipelineStatus.NEW,)
    )

    # 2) Get claimable worklist (respects human gate + lease expiration)
    worklist: list[tuple[str, int]] = get_claimable_worklist(
        stage=PipelineStage.JOB_URLS,
        status=statuses,
        max_rows=max_concurrent_tasks * 4 or 1000,
    )

    # 3) Optional subset filter
    if filter_keys:
        filter_set = set(filter_keys)
        worklist = [(u, it) for (u, it) in worklist if u in filter_set]

    if not worklist:
        logger.info("📭 No claimable (url, iteration) at stage 'job_urls'.")
        return

    # 4) Create a stable worker identity for this run
    worker_id = generate_worker_id(prefix="job_postings")

    logger.info(
        "🚀 Starting job_postings pipeline | %d items | worker_id=%s | stage=job_urls",
        len(worklist),
        worker_id,
    )

    # 5) Launch the batch
    tasks = await process_job_postings_batch_async_fsm(
        items=worklist,
        worker_id=worker_id,
        lease_minutes=lease_minutes,
        llm_provider=llm_provider,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        no_of_concurrent_workers=max_concurrent_tasks,
    )

    await asyncio.gather(*tasks)
    logger.info("✅ Finished job_postings FSM pipeline.")
