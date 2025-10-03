"""
fsm/state_map_config.py

Centralized config for managing the `state_map` JSON field
in the `pipeline_control` table.

This module defines:
- Valid stages (from PipelineStage enum)
- Valid status values per stage (PipelineStatus)
- Default state_map initializer
- Helper functions for validation and patching

⚠️ Note:
    This module only manages per-stage FSM statuses (PipelineStatus).
    The higher-level lifecycle field `process_status`
    (NEW, RUNNING, COMPLETED, SKIPPED) is tracked separately
    at the row level in `pipeline_control` and is intentionally
    out of scope here.
"""

from typing import Dict
from db_io.pipeline_enums import PipelineStage, PipelineStatus


def get_valid_stage_keys() -> list[str]:
    """
    Return all valid stage names used in the state_map.

    These come from PipelineStage enum values.
    """
    return [stage.value for stage in PipelineStage]


def get_default_state_map() -> Dict[str, str]:
    """
    Return a default state_map with all stages initialized to 'new'.
    """
    return {stage: PipelineStatus.NEW.value for stage in get_valid_stage_keys()}


def is_valid_state_value(value: str) -> bool:
    """
    Check if a given value is a valid state status.
    """
    return value in {
        e.value for e in PipelineStatus
    }  # can also use ._value2member_map_ (internal implementation)


def validate_state_map(state_map: dict) -> bool:
    """
    Validate that a given state_map has only known keys and valid values.

    Returns True if valid, False otherwise.
    """
    for key, val in state_map.items():
        if key not in get_valid_stage_keys():
            return False
        if not is_valid_state_value(val):
            return False
    return True


def patch_state_map(
    state_map: dict,
    stage: str,
    new_value: str,
    allow_new_keys: bool = False,
) -> dict:
    """
    Update a specific stage in the state_map with a new value.

    Args:
        state_map (dict): The original state_map
        stage (str): The pipeline stage key to update
        new_value (str): The new status value
        allow_new_keys (bool): Whether to allow adding new keys not in the enum

    Returns:
        dict: Updated state_map
    """
    if not is_valid_state_value(new_value):
        raise ValueError(f"Invalid status value: {new_value}")
    if not allow_new_keys and stage not in get_valid_stage_keys():
        raise KeyError(f"Invalid stage name: {stage}")
    state_map[stage] = new_value
    return state_map
