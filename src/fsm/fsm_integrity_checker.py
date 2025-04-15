from db_io.db_utils import get_urls_by_stage_and_status
from db_io.schema_definitions import PipelineStage
from fsm.pipeline_fsm import PipelineFSM, get_transitions
from db_io.state_sync import load_pipeline_state


def validate_fsm_integrity():
    """Ensure all recorded FSM transitions comply with FSM rules"""
    allowed_transitions = {
        (t["source"], t["dest"]) for t in get_transitions(PipelineFSM.STAGES)
    }

    issues = []
    for stage in PipelineStage.list():
        urls = get_urls_by_stage_and_status(stage=stage)
        for url in urls:
            state = load_pipeline_state(url)
            if (
                state
                and state.last_stage
                and (state.last_stage, stage) not in allowed_transitions
            ):
                issues.append((url, state.last_stage, stage))

    if issues:
        print("FSM Integrity Issues Found:")
        for issue in issues:
            print(f"URL: {issue[0]}, Invalid transition: {issue[1]} -> {issue[2]}")
    else:
        print("No FSM integrity issues detected.")
