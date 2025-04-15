from db_io.db_utils import get_stage_progress_counts, get_recent_urls


def print_pipeline_summary():
    """Print a quick summary of pipeline stages/statuses"""
    summary_df = get_stage_progress_counts()
    print(summary_df)


def recent_activities(limit: int = 10):
    """List recent pipeline activities (state changes, new URLs)"""
    recent_df = get_recent_urls(limit)
    print(recent_df)
