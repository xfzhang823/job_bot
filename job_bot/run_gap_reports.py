"""
run_gap_reports.py

Generate + persist requirement gap reports (CSV/JSON/TXT) for a given job URL.
"""

from __future__ import annotations

import logging

from job_bot.config.project_config import GAP_REPORTS_DIR
from job_bot.db_io.pipeline_enums import Version
from job_bot.utils.gap_report import (
    GapReportConfig,
    GapThresholds,
    generate_gap_report,
    persist_gap_report,
)

logger = logging.getLogger(__name__)


def run_one_url_gap_report(*, url: str, version: Version, iteration: int = 0) -> None:
    # 1) Build config (THIS is where the URL filter is set)
    cfg = GapReportConfig(
        url=url,
        version=version.value,
        iteration=iteration,
        resp_llm_provider=None,
        resp_model_id=None,
        thresholds=GapThresholds(strong=0.45, medium=0.35),
    )

    # 2) Generate report (in-memory)
    df_best, summary, text = generate_gap_report(cfg, max_rows=25)

    # 3) Log the text report
    logger.info("\n%s", text)

    # 4) Persist artifacts (CSV + JSON + TXT)
    csv_path, json_path, txt_path = persist_gap_report(
        df_best=df_best,
        summary=summary,
        report_text=text,
        output_dir=GAP_REPORTS_DIR,
        url=cfg.url,
        version=cfg.version,  # include in filenames + JSON metadata
        iteration=cfg.iteration,
    )

    logger.info("Saved:\n- %s\n- %s\n- %s", csv_path, json_path, txt_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    url = "https://searchjobs.libertymutualgroup.com/careers/job/618514434537"
    run_one_url_gap_report(url=url, version=Version.ORIGINAL, iteration=0)


if __name__ == "__main__":
    main()
