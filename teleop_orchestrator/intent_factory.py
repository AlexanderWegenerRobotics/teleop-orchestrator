"""Config-driven construction of the trained intent model teleop_intent
ships, per system.yaml's modules.intent block. No adapter class needed:
IntentModel.reset()/step(SensorFrame)->IntentOutput already satisfies
contracts.Module structurally (see teleop-intent/models/base.py's
docstring) -- this is purely "pick the right class and load its checkpoint."

With modules.intent.task_prior set, the model is wrapped in
teleop_intent.models.composed.TaskCorrectedIntentModel: the perceptual
model's target posterior is multiplied by p(target | filled slots)^prior_beta
(see teleop-intent/models/task_prior). The wrapper has the same interface, so
nothing downstream changes; the base model never sees the prior.
"""

from __future__ import annotations

import importlib

_FAMILIES = {
    "hmm": ("teleop_intent.models.hmm.model", "HMMIntentModel"),
    "gru": ("teleop_intent.models.gru.model", "GRUIntentModel"),
    "transformer": ("teleop_intent.models.transformer.model", "TransformerIntentModel"),
}


def build_intent_model(family: str, checkpoint: str, task_prior: str | None = None, prior_beta: float = 0.5):
    """Instantiates the configured intent model family, loads its checkpoint, and
    wraps it with the task prior when one is configured."""
    if family not in _FAMILIES:
        raise ValueError(f"unknown intent model family {family!r}; choose one of {list(_FAMILIES)}")
    module_name, class_name = _FAMILIES[family]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls()
    model.load(checkpoint)
    if task_prior:
        TaskPrior = importlib.import_module("teleop_intent.models.task_prior.model").TaskPrior
        TaskCorrectedIntentModel = importlib.import_module("teleop_intent.models.composed").TaskCorrectedIntentModel
        model = TaskCorrectedIntentModel(model, TaskPrior.load(task_prior), beta=float(prior_beta))
    return model


def build_intent_from_config(icfg: dict):
    """build_intent_model from a modules.intent block (family, checkpoint, task_prior, prior_beta)."""
    return build_intent_model(icfg["family"], icfg["checkpoint"],
                              task_prior=icfg.get("task_prior"), prior_beta=icfg.get("prior_beta", 0.5))
