""" "TBA"""

from db_io.state_sync import persist_pipeline_state_to_duckdb, load_pipeline_state
from fsm.pipeline_fsm_manager import PipelineFSMManager
from db_io.schema_definitions import PipelineStage, PipelineStatus
from models.duckdb_table_models import PipelineState


class PipelineControl:
    def __init__(self):
        self.fsm_manager = PipelineFSMManager()

    def initialize_urls(
        self, urls: list[str], iteration: int = 0, llm_provider: str = "openai"
    ):
        """Initialize FSM state for new URLs"""
        for url in urls:
            state = load_pipeline_state(url)
            if not state:
                new_state = PipelineState(
                    url=url,
                    iteration=iteration,
                    llm_provider=llm_provider,
                    last_stage=PipelineStage.JOB_URLS.value,
                    status=PipelineStatus.NEW,
                )
                persist_pipeline_state_to_duckdb(new_state)

    def bulk_update_status(self, urls: list[str], status: PipelineStatus):
        """Bulk update the status of multiple URLs"""
        for url in urls:
            state = load_pipeline_state(url)
            if state:
                state.status = status
                persist_pipeline_state_to_duckdb(state)

    def step_fsm(self, url: str):
        """Explicitly advance FSM for a URL"""
        fsm = self.fsm_manager.get_fsm(url)
        fsm.step()
