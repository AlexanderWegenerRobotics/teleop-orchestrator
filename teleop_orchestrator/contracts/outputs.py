"""Output payloads emitted by each module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NULL_TARGET = -1


class Phase:
    """Intended-action classes (what the operator is trying to do, not observed state)."""
    IDLE = 0
    UNDECIDED = 1
    APPROACH = 2
    GRASP = 3
    TRANSPORT = 4
    PLACE = 5
    N_CLASSES = 6
    NAMES = {0: "idle", 1: "undecided", 2: "approach", 3: "grasp", 4: "transport", 5: "place"}


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) of a probability vector, ignoring zero entries."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


@dataclass
class ArmIntent:
    """Per-arm intent posteriors from the intent module."""

    phase_posterior: np.ndarray                     # [Phase.N_CLASSES]
    target_posterior: np.ndarray                    # [n_candidates]
    location_posterior: Optional[np.ndarray] = None # [n_candidates] destination lookahead

    def top_phase(self) -> int:
        """Most likely phase class."""
        return int(np.argmax(self.phase_posterior))

    def top_target(self) -> int:
        """Index of the most likely target candidate."""
        return int(np.argmax(self.target_posterior))

    def target_entropy(self) -> float:
        """Entropy of the target posterior; the uncertainty signal the nudge gate reads."""
        return _entropy(self.target_posterior)

    def is_confident(self, max_target_entropy: float) -> bool:
        """Whether the target posterior is peaked enough to act on."""
        return self.target_entropy() <= max_target_entropy


@dataclass
class IntentOutput:
    """Intent module output for one tick: independent per-arm intent."""

    left: ArmIntent
    right: ArmIntent
    extras: dict = field(default_factory=dict)

    def arm(self, name: str) -> ArmIntent:
        """Returns the ArmIntent for 'left' or 'right'."""
        return self.left if name == "left" else self.right


@dataclass
class ActionOutput:
    """Imitation-learning module output: the predicted next action, per arm."""

    ee_pose: dict[str, np.ndarray]      # "arm_left"/"arm_right" -> commanded pose
    gripper: dict[str, float]           # "arm_left"/"arm_right" -> commanded gripper
    extras: dict = field(default_factory=dict)


@dataclass
class AssistOutput:
    """Assistance module decision: what to surface to the operator this tick."""

    active: bool                        # whether any assistance is offered now
    kind: str = ""                      # e.g. "target_hint", "arm_hint", "force_cue"
    target: int = NULL_TARGET           # candidate the hint refers to, if any
    arm: str = ""                       # "left"/"right"/"" if not arm-specific
    payload: dict = field(default_factory=dict)