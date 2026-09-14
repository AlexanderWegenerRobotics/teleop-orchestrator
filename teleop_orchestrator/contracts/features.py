"""Shared intent-candidate feature extraction: the one place raw episode.hdf5
intent-log columns become SensorFrame candidate/global features. ReplaySource
and the intent training loader both call this, so a model never sees
different inputs at train time vs runtime.

Layout facts baked in here (verified against intention_buffer.cpp and a real
episode's scene ground truth, not assumed):
  - slot_type_/slot_px_u_/slot_px_v_ arrays are joint-indexed: index 0 = ee_left,
    1 = ee_right, 2.. = pick/place candidates (objects then bins).
  - slot_dist_ is INTERLEAVED per pick/place candidate: [left_0, right_0,
    left_1, right_1, ...], covering only pick/place slots (never the 2 EE
    slots themselves).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np

# Raw SlotType ids written by the C++ intention pipeline (intention_sample.hpp).
SLOT_EE_LEFT = 0
SLOT_EE_RIGHT = 1
SLOT_PICK_OBJ = 2
SLOT_PLACE = 3
N_EE_SLOTS = 2  # ee_left, ee_right always occupy joint-slot indices 0 and 1

# Candidate-space type ids (end-effector is excluded from candidates entirely —
# see SensorFrame's docstring: "end-effector is NOT a candidate").
CAND_OBJECT = 0
CAND_BIN = 1
CAND_UNKNOWN = -1
_RAW_TO_CAND_TYPE = {SLOT_PICK_OBJ: CAND_OBJECT, SLOT_PLACE: CAND_BIN}

# Columns that are the live filter's own posterior over targets and must never
# enter a model's input: a learned model could otherwise copy the baseline's
# answer instead of learning to predict it. slot_belief_ is the legacy
# (pre schema-v2) name; tgt_belief_ is the current split-belief name.
LEAKAGE_PREFIXES = ("slot_belief_", "tgt_belief_")


def _native_camera_dims(obs: Optional[h5py.Group]) -> tuple[float, float]:
    """Returns (native_width, native_height) of the primary camera used to
    record gaze/slot pixel positions, from the stored image dataset attrs.

    Falls back to 1280x960 (the rig's single-camera resolution at the time
    all current schema<=2 episodes were recorded) if obs/images/attrs are
    unavailable, so very old conversions don't hard-fail -- but a real
    episode should always have these.
    """
    if obs is not None:
        images = obs.get("images")
        if images is not None:
            for cam in ("head_cam_left", "head_cam_right"):
                if cam in images and "native_width" in images[cam].attrs:
                    return float(images[cam].attrs["native_width"]), float(images[cam].attrs["native_height"])
    return 1280.0, 960.0


def _slot_columns(group: h5py.Group, prefix: str) -> list[str]:
    """Returns the slot dataset names under `prefix`, ordered by their integer index."""
    cols = []
    for k in group.keys():
        if k.startswith(prefix) and k[len(prefix):].isdigit():
            cols.append((int(k[len(prefix):]), k))
    return [name for _, name in sorted(cols)]


def _stack(group: h5py.Group, names: list[str]) -> np.ndarray:
    """Loads the named 1-D datasets into a [T, len(names)] array in memory."""
    return np.stack([group[n][:] for n in names], axis=1) if names else np.empty((0, 0))


@dataclass
class IntentArrays:
    """One episode's intent-log columns, loaded once and reused per timestep."""

    n_pickplace: int             # candidate slot capacity, EE slots excluded
    types: np.ndarray            # [T, n_pickplace] raw SlotType (2=object, 3=bin)
    px_u: np.ndarray             # [T, n_pickplace]
    px_v: np.ndarray             # [T, n_pickplace]
    dist: np.ndarray             # [T, n_pickplace, 2] (dist_to_left_ee, dist_to_right_ee)
    n_slots: np.ndarray          # [T] total active joint slots this frame (incl. the 2 EE slots)
    gaze: np.ndarray             # [T, 2] (x, y)
    gaze_valid: np.ndarray       # [T] bool
    gripper: np.ndarray          # [T, 2] (left, right) commanded/logged gripper value
    ee_pos: dict[str, np.ndarray]        # "left"/"right" -> [T, 3] world position
    ee_belief: Optional[np.ndarray]      # [T, 2] (left, right) gaze-on-arm attention, or None
    pos: np.ndarray               # [T, n_pickplace, 3] candidate world (x,y,z); NaN where
                                   # unavailable (pre-backfill episodes, or a padded/empty slot).
                                   # Sim-privileged (scene.csv), see candidate_world_pos_at.


def load_intent_arrays(intent: h5py.Group, obs: Optional[h5py.Group] = None) -> IntentArrays:
    """Loads and re-indexes one episode's intent-log columns.

    End-effector slots (joint indices 0, 1) are split out here: they are
    scene-level ee-attention signals for global_features, not manipulation
    candidates. Any slot_belief_/tgt_belief_ column is never read, even if
    present, so it can't leak into a feature by accident later.

    `obs` (the parent "observations" group, containing "images") is optional
    but should always be passed by real callers -- it's needed to reconcile
    gaze_px_x/y against slot_px_u/v into the same coordinate space (see the
    unit-reconciliation block below); without it, legacy (schema<=2)
    episodes fall back to an assumed 1280x960 native camera resolution.
    """
    type_cols = _slot_columns(intent, "slot_type_")
    n_total = len(type_cols)
    n_pickplace = max(0, n_total - N_EE_SLOTS)

    types = _stack(intent, type_cols[N_EE_SLOTS:N_EE_SLOTS + n_pickplace])

    px_u_cols = _slot_columns(intent, "slot_px_u_")[N_EE_SLOTS:N_EE_SLOTS + n_pickplace]
    px_v_cols = _slot_columns(intent, "slot_px_v_")[N_EE_SLOTS:N_EE_SLOTS + n_pickplace]
    px_u = _stack(intent, px_u_cols)
    px_v = _stack(intent, px_v_cols)

    dist_cols = _slot_columns(intent, "slot_dist_")
    flat_dist = _stack(intent, dist_cols)  # [T, 2 * capacity], interleaved [l0, r0, l1, r1, ...]
    T = flat_dist.shape[0] if flat_dist.size else int(intent["n_slots"].shape[0])
    n_pairs_avail = flat_dist.shape[1] // 2 if flat_dist.size else 0
    n_use = min(n_pickplace, n_pairs_avail)
    dist = np.zeros((T, n_pickplace, 2), dtype=np.float64)
    if n_use:
        dist[:, :n_use, :] = flat_dist[:, :2 * n_use].reshape(T, n_use, 2)

    # Candidate world position, if this episode has been converted/backfilled
    # with it (see backfill_candidate_position.py / episode_to_hdf5.py).
    # Joint-slot indexed same as slot_px_u_/slot_px_v_, so the same
    # N_EE_SLOTS offset and capacity trimming applies. Absent entirely on
    # older episodes -- left as NaN rather than 0.0 so a caller can't
    # mistake "no data" for "position (0,0,0)".
    pos_x_cols = _slot_columns(intent, "slot_pos_x_")[N_EE_SLOTS:N_EE_SLOTS + n_pickplace]
    pos_y_cols = _slot_columns(intent, "slot_pos_y_")[N_EE_SLOTS:N_EE_SLOTS + n_pickplace]
    pos_z_cols = _slot_columns(intent, "slot_pos_z_")[N_EE_SLOTS:N_EE_SLOTS + n_pickplace]
    n_pos = min(len(pos_x_cols), len(pos_y_cols), len(pos_z_cols))
    pos = np.full((T, n_pickplace, 3), np.nan, dtype=np.float64)
    if n_pos:
        pos[:, :n_pos, 0] = _stack(intent, pos_x_cols[:n_pos])
        pos[:, :n_pos, 1] = _stack(intent, pos_y_cols[:n_pos])
        pos[:, :n_pos, 2] = _stack(intent, pos_z_cols[:n_pos])

    n_slots = intent["n_slots"][:].astype(int)
    gaze = np.stack([intent["gaze_px_x"][:], intent["gaze_px_y"][:]], axis=1)
    gaze_valid = intent["gaze_valid"][:] > 0.5

    # Reconcile gaze and slot pixel positions into one shared coordinate
    # space, so a model's gaze-to-candidate distance is geometrically
    # meaningful. Schema v3+ ("normalized_ray"): gaze_px_x/y and slot_px_u/v
    # are already the same pinhole-ray convention against the same camera --
    # nothing to do. Schema <=2: gaze_px_x/y (when px_normalized) are a 0-1
    # fraction of the primary camera's frame, while slot_px_u/v are raw
    # native pixels -- two different unit systems that used to be fed
    # straight through to candidate_features_at/global_features_at
    # unreconciled (only the viz code in teleop-intent/common.py did this
    # rescale, for display only). That meant TargetStickyFilter's gaze<->
    # candidate distance was comparing a ~0-1 fraction against ~1000px raw
    # pixels -- effectively noise, however well its sigma was fit. Rescale
    # both into "fraction of native single-camera frame" here so every
    # consumer (ReplaySource at runtime, the training loader) sees
    # consistent units.
    if intent.attrs.get("gaze_units") != "normalized_ray":
        native_w, native_h = _native_camera_dims(obs)
        if bool(intent.attrs.get("px_normalized", False)):
            pass  # gaze already a 0-1 fraction of the native camera frame
        else:
            ref_w = float(intent.attrs.get("gaze_px_ref_width", native_w))
            ref_h = float(intent.attrs.get("gaze_px_ref_height", native_h))
            gaze = gaze / np.array([ref_w, ref_h])
        if native_w > 0 and native_h > 0:
            px_u = px_u / native_w
            px_v = px_v / native_h
    gripper = np.stack([intent["gripper_left"][:], intent["gripper_right"][:]], axis=1)
    ee_pos = {
        "left": np.stack([intent["ee_left_x"][:], intent["ee_left_y"][:], intent["ee_left_z"][:]], axis=1),
        "right": np.stack([intent["ee_right_x"][:], intent["ee_right_y"][:], intent["ee_right_z"][:]], axis=1),
    }
    ee_belief = None
    if "ee_belief_ee_left" in intent and "ee_belief_ee_right" in intent:
        ee_belief = np.stack([intent["ee_belief_ee_left"][:], intent["ee_belief_ee_right"][:]], axis=1)

    return IntentArrays(
        n_pickplace=n_pickplace, types=types, px_u=px_u, px_v=px_v, dist=dist,
        n_slots=n_slots, gaze=gaze, gaze_valid=gaze_valid, gripper=gripper,
        ee_pos=ee_pos, ee_belief=ee_belief, pos=pos,
    )


def candidate_features_at(arr: IntentArrays, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (features[n_pickplace, 4], mask[n_pickplace], types[n_pickplace]) at timestep t.

    Features are (px_u, px_v, dist_to_left_ee, dist_to_right_ee). The
    end-effectors themselves are never included as candidates.
    """
    n_real = max(0, int(arr.n_slots[t]) - N_EE_SLOTS)
    mask = np.arange(arr.n_pickplace) < n_real
    features = np.concatenate([arr.px_u[t][:, None], arr.px_v[t][:, None], arr.dist[t]], axis=1)
    types = np.array([_RAW_TO_CAND_TYPE.get(int(v), CAND_UNKNOWN) for v in arr.types[t]], dtype=np.int64)
    return features, mask, types


def candidate_world_pos_at(arr: IntentArrays, t: int) -> np.ndarray:
    """Returns candidate world positions [n_pickplace, 3] (x, y, z) at
    timestep t -- sim-privileged ground truth from scene.csv, NOT part of
    the deployment-stable candidate_features_at contract (see
    SensorFrame.candidate_world_pos's docstring for why). NaN rows mean no
    data (older, non-backfilled episode, or an empty/padded slot); callers
    must treat NaN as "no evidence available", never as zero.
    """
    return arr.pos[t]


def _decode(v) -> str:
    """Decodes an hdf5 string cell (bytes or str) to a plain str."""
    return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)


def candidate_names(intent: h5py.Group) -> list[str]:
    """Returns pick/place candidate names in the same index order as
    candidate_features_at/candidate_mask (end-effector slots excluded).

    Only meaningful offline (labeling, training-label alignment, playback
    overlays) — a deployed model consumes candidate indices, never names, so
    this is never on the runtime path.
    """
    name_cols = _slot_columns(intent, "slot_name_")[N_EE_SLOTS:]
    return [_decode(intent[c][0]) for c in name_cols]


def global_features_at(arr: IntentArrays, t: int) -> np.ndarray:
    """Returns the scene-level (non-candidate) feature vector at timestep t:
    gaze (x, y), gaze_valid, gripper (L, R), ee position (L, R), and per-arm
    ee-attention belief (L, R) if the episode has been recomputed with it.
    """
    parts = [
        arr.gaze[t], np.array([float(arr.gaze_valid[t])]), arr.gripper[t],
        arr.ee_pos["left"][t], arr.ee_pos["right"][t],
    ]
    parts.append(arr.ee_belief[t] if arr.ee_belief is not None else np.zeros(2))
    return np.concatenate(parts).astype(np.float64)
