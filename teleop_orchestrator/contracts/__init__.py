"""Shared runtime contracts. Leaf package: imports nothing from the parent orchestrator."""

from .frame import SensorFrame, CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
from .module import Module, SensorModule
from .outputs import (
    NULL_TARGET,
    Phase,
    ArmIntent,
    IntentOutput,
    ActionOutput,
    AssistOutput,
)
from .features import (
    IntentArrays,
    load_intent_arrays,
    candidate_features_at,
    candidate_world_pos_at,
    global_features_at,
    candidate_names,
)

__all__ = [
    "SensorFrame",
    "CANDIDATE_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "Module",
    "SensorModule",
    "NULL_TARGET",
    "Phase",
    "ArmIntent",
    "IntentOutput",
    "ActionOutput",
    "AssistOutput",
    "IntentArrays",
    "load_intent_arrays",
    "candidate_features_at",
    "candidate_world_pos_at",
    "global_features_at",
    "candidate_names",
]