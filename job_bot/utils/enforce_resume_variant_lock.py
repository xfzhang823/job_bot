"""
utils/enforce_resume_variant_lock.py
"""

from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.config.resume_variant import ResumeVariant


def enforce_resume_variant_lock(
    *,
    url: str,
    resume_variant: ResumeVariant,
    touch_updated_at: bool = True,
) -> None:
    """
    Enforce a **hard, URL-global lock** between a job posting URL and a single
    resume variant.

    This function guarantees that **all resume-derived pipelines** for a given
    URL operate against the same resume variant (e.g., MI_STRATEGY,
    AI_ARCHITECT). It is the single authoritative enforcement point for this
    invariant.

    Source of truth:
        Table: `url_resume_variant`
        Primary key: (url)

    Behavior:
        1) If no row exists for `url`:
           • Insert a new binding (url → resume_variant).
        2) If a row exists and `resume_variant` matches:
           • Allow execution to continue.
           • Optionally touch `updated_at`.
        3) If a row exists and `resume_variant` mismatches:
           • Raise RuntimeError (HARD FAIL).

    Failure semantics:
        • This function intentionally raises on mismatch.
        • Callers MUST catch the exception and finalize/release any leases
          already acquired (e.g., via `finalize_one_row_in_pipeline_control`).
        • No resume-derived writes should occur after a mismatch.

    Intended call site:
        • After `try_claim_one(...)` succeeds (lease acquired).
        • Before any resume JSON is read or any resume-derived tables are written.

    Design notes:
        • The lock is URL-global (not iteration-scoped).
        • Iteration represents reruns, not configuration changes.
        • Variant changes require an explicit manual reset of the lock table.
        • This prevents silent cross-variant contamination of downstream data.

    Args:
        url (str):
            Canonical job posting URL (unit of work).

        resume_variant (ResumeVariant):
            Enum value representing the resume variant selected for this run.

        touch_updated_at (bool):
            If True, updates `updated_at` when an existing matching lock is
            observed. Defaults to True.

    Raises:
        RuntimeError:
            If the URL is already bound to a different resume variant.
    """

    con = get_db_connection()
    try:
        incoming = resume_variant.value

        row = con.execute(
            """
            SELECT resume_variant
            FROM url_resume_variant
            WHERE url = ?
            """,
            [url],
        ).fetchone()

        # bind if missing
        if row is None:
            con.execute(
                """
                INSERT INTO url_resume_variant (url, resume_variant, created_at, updated_at)
                VALUES (?, ?, now(), CASE WHEN ? THEN now() ELSE NULL END)
                """,
                [url, incoming, touch_updated_at],
            )
            return

        existing = row[0]

        # mismatch → hard fail
        if existing != incoming:
            raise RuntimeError(
                f"❌ Resume variant mismatch for URL={url}\n"
                f"   existing={existing}\n"
                f"   incoming={incoming}\n"
                f"   This is a hard lock violation."
            )

        # match → optionally touch
        if touch_updated_at:
            con.execute(
                """
                UPDATE url_resume_variant
                SET updated_at = now()
                WHERE url = ?
                """,
                [url],
            )

    finally:
        con.close()
