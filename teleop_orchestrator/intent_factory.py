"""Config-driven construction of the trained intent model teleop_intent
ships, per system.yaml's modules.intent block. No adapter class needed:
IntentModel.reset()/step(SensorFrame)->IntentOutput already satisfies
contracts.Module structurally (see teleop-intent/models/base.py's
docstring) -- this is purely "pick the right class and load its checkpoint."
"""

from __future__ import annotations

import importlib

_FAMILIES = {
    "hmm": ("teleop_intent.models.hmm.model", "HMMIntentModel"),
    "gru": ("teleop_intent.models.gru.model", "GRUIntentModel"),
    "transformer": ("teleop_intent.models.transformer.model", "TransformerIntentModel"),
}


def build_intent_model(family: str, checkpoint: str):
    """Instantiates the configured intent model family and loads its checkpoint."""
    if family not in _FAMILIES:
        raise ValueError(f"unknown intent model family {family!r}; choose one of {list(_FAMILIES)}")
    module_name, class_name = _FAMILIES[family]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls()
    model.load(checkpoint)
    return model
