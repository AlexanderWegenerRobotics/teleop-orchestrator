"""Canonical per-tick observation shared by all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SensorFrame:
    """One tick of synchronized observation, the live analog of an episode.hdf5 grid row."""

    timestamp_ns: int
    frame_id: int

    # Per-candidate features (objects + bins; end-effector is NOT a candidate).
    candidate_features: np.ndarray      # [n_candidates, n_candidate_features]
    candidate_mask: np.ndarray          # [n_candidates] bool, True = real candidate
    candidate_types: np.ndarray         # [n_candidates] int, semantic type per slot

    # Scene-level signals not tied to a candidate (gripper, wrench, ee motion,
    # per-arm ee-attention, aggregate object/bin attention mass, ...).
    global_features: np.ndarray         # [n_global_features]

    # Proprioception, per arm (joint + ee state). Keyed "arm_left"/"arm_right".
    proprio: dict[str, np.ndarray] = field(default_factory=dict)

    # Camera views, keyed by name; present at runtime and for the IL module,
    # absent for gaze-only intent inference. Left as raw arrays here.
    images: dict[str, np.ndarray] = field(default_factory=dict)

    gaze_valid: bool = True
    engaged: bool = True                # both arms in a training/active state

    @property
    def n_candidates(self) -> int:
        """Number of candidate slots this tick, including padded ones."""
        return int(self.candidate_features.shape[0])

    @property
    def usable(self) -> bool:
        """Whether this tick is a valid sample for intent (engaged with valid gaze)."""
        return self.engaged and self.gaze_valid