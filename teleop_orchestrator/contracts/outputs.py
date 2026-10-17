"""Output payloads emitted by each module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NULL_TARGET = -1


class Phase:
    """Intended-action classes (what the operator is trying to do, not observed state)."""
    IDLE = 0
    APPROACH = 1
    GRASP = 2
    TRANSPORT = 3
    PLACE = 4
    INSERT = 5
    CORRECT = 6
    N_CLASSES = 7
    NAMES = {0: "idle", 1: "approach", 2: "grasp", 3: "transport", 4: "place", 5: "insert", 6: "correct"}

    @staticmethod
    def pad(posterior: np.ndarray) -> np.ndarray:
        """Zero-pads a posterior from a model with fewer phases to N_CLASSES."""
        p = np.asarray(posterior, dtype=np.float64)
        if p.shape[-1] >= Phase.N_CLASSES:
            return p
        return np.concatenate([p, np.zeros(p.shape[:-1] + (Phase.N_CLASSES - p.shape[-1],))], axis=-1)


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

    def phase_entropy(self) -> float:
        """Entropy of the phase posterior."""
        return _entropy(self.phase_posterior)

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
class NudgeOutput:
    """Nudge engine decision for one tick: the cue commands issued and the proposals it held back."""

    issued: list = field(default_factory=list)
    suppressed: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        """Whether any cue went out this tick."""
        return len(self.issued) > 0


AssistOutput = NudgeOutput