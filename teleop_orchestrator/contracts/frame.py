"""Canonical per-tick observation shared by all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Canonical column order for candidate_features (see candidate_features_at in
# contracts.features) and global_features (see ReplaySource._global_features).
# Anything that needs a specific field looks it up by name here instead of a
# magic index, so producer and consumer can never silently drift apart on
# what column N means.
CANDIDATE_FEATURE_NAMES = ["px_u", "px_v", "dist_left", "dist_right"]

GLOBAL_FEATURE_NAMES = [
    "gaze_x", "gaze_y", "gaze_valid",
    "gripper_left", "gripper_right",
    "ee_left_x", "ee_left_y", "ee_left_z",
    "ee_right_x", "ee_right_y", "ee_right_z",
    "ee_belief_left", "ee_belief_right",
    "gripper_width_left", "gripper_width_right",
    "fext_left_fx", "fext_left_fy", "fext_left_fz",
    "fext_left_tx", "fext_left_ty", "fext_left_tz",
    "fext_right_fx", "fext_right_fy", "fext_right_fz",
    "fext_right_tx", "fext_right_ty", "fext_right_tz",
]


@dataclass
class SensorFrame:
    """One tick of synchronized observation, the live analog of an episode.hdf5 grid row."""

    timestamp_ns: int
    frame_id: int

    # Per-candidate features (objects + bins; end-effector is NOT a candidate).
    # Columns are CANDIDATE_FEATURE_NAMES, in that order.
    candidate_features: np.ndarray      # [n_candidates, n_candidate_features]
    candidate_mask: np.ndarray          # [n_candidates] bool, True = real candidate
    candidate_types: np.ndarray         # [n_candidates] int, semantic type per slot

    # Scene-level signals not tied to a candidate (gripper, wrench, ee motion,
    # per-arm ee-attention, ...). Fields are GLOBAL_FEATURE_NAMES, in that order.
    global_features: np.ndarray         # [n_global_features]

    # Proprioception, per arm (joint + ee state). Keyed "arm_left"/"arm_right".
    proprio: dict[str, np.ndarray] = field(default_factory=dict)

    # Camera views, keyed by name; present at runtime and for the IL module,
    # absent for gaze-only intent inference. Left as raw arrays here.
    images: dict[str, np.ndarray] = field(default_factory=dict)

    gaze_valid: bool = True
    engaged: bool = True                # both arms in a training/active state

    # Which command channel the avatar is letting move each arm this tick, keyed
    # "arm_left"/"arm_right" (values: live.object_source.AUTHORITY_*). Per arm,
    # because the operator can take one hand back while the policy keeps the
    # other -- so this is a dict rather than a scalar, and every consumer has to
    # decide per arm rather than for the robot as a whole.
    #
    # An arm missing from the dict means the avatar did not say, NOT that it is
    # unclaimed: read it with .get(side, AUTHORITY_UNSET) so an avatar without
    # the field behaves exactly as it did before authority existed.
    authority: dict[str, int] = field(default_factory=dict)

    # Measured head pan/tilt this tick, in radians -- the same two joints, in
    # the same order, as the head's own q vector.
    #
    # Carried on the frame for the same reason authority is. The head's UDP
    # channel is point-to-point on the avatar side, so whenever the VR
    # interface is running it owns that port and this process's HeadChannel
    # receives nothing at all; system.yaml says as much next to
    # network.head.receive_port. LiveSource resolves the two sources once per
    # tick (channel first, scene publish as fallback) and everything downstream
    # reads the answer here, rather than repeating that fallback and getting it
    # wrong -- which PolicyModule did, skipping every tick on a head state that
    # was never going to arrive.
    #
    # 0.0 is a real, reachable pose (neck centred), not a sentinel: an avatar
    # predating head_pan/head_tilt on SceneObjectsMsg looks exactly like one
    # looking straight ahead.
    head_pan: float = 0.0
    head_tilt: float = 0.0

    # Per-arm grasp confirmation (ArmControl::updateGraspConfirmation, real
    # sensor-derived signal, not privileged sim state -- gripper width settled
    # at a value consistent with holding something, while a grasp was
    # commanded). Keyed "left"/"right"; a side absent from this dict means the
    # signal wasn't available for this episode (older logs, backfilled
    # separately or not at all), not that nothing is held -- callers needing
    # a boolean must handle the absent case explicitly rather than assuming False.
    grasp_confirmed: dict[str, bool] = field(default_factory=dict)

    # Candidate world position (x, y, z), sim-privileged ground truth from
    # scene.csv (see backfill_candidate_position.py / episode_to_hdf5.py).
    # Deliberately kept OUT of candidate_features/CANDIDATE_FEATURE_NAMES:
    # those columns are meant to stay deployment-stable (vision-derived,
    # eventually real on hardware too), while this one has no real-hardware
    # equivalent yet and would need a different proxy at deployment. None
    # means unavailable for this episode (pre-backfill); rows may still be
    # individually NaN even when the array is present (a slot with no
    # matching scene.csv column, or padding beyond n_candidates).
    candidate_world_pos: Optional[np.ndarray] = None  # [n_candidates, 3] or None

    @property
    def n_candidates(self) -> int:
        """Number of candidate slots this tick, including padded ones."""
        return int(self.candidate_features.shape[0])

    @property
    def usable(self) -> bool:
        """Whether this tick is a valid sample for intent (engaged with valid gaze)."""
        return self.engaged and self.gaze_valid