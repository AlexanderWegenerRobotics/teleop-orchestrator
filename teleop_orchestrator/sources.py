"""Replay an episode.hdf5 as a stream of SensorFrames (offline driving of the orchestrator)."""

from __future__ import annotations

import time
from typing import Iterator, Optional

import h5py
import numpy as np

from .contracts import SensorFrame, GLOBAL_FEATURE_NAMES
from .contracts.features import (
    IntentArrays, load_intent_arrays, candidate_features_at, candidate_world_pos_at, global_features_at,
)

ENGAGED_DEFAULT = (4,)


class ReplaySource:
    """Yields one SensorFrame per grid timestep from a converted episode.hdf5.

    Telemetry/intent columns are loaded fully into memory (small); images are
    read lazily per frame (large). Candidate/global feature assembly lives in
    contracts.features, shared with the intent training loader so a model
    never sees different inputs at train time vs runtime.
    """

    def __init__(self, path: str, rate_hz: Optional[float] = None,
                 load_images: bool = True, engaged_states=ENGAGED_DEFAULT):
        self.path = path
        self.rate_hz = rate_hz
        self.load_images = load_images
        self.engaged_states = tuple(engaged_states)
        self._f = h5py.File(path, "r")
        self._prime()

    def _prime(self) -> None:
        """Loads grid, intent, and proprio columns into memory and precomputes masks."""
        obs = self._f["observations"]
        self._ts = obs["timestamp_ns"][:]
        self._frame_id = obs["frame_id"][:] if "frame_id" in obs else np.arange(len(self._ts))
        self._T = len(self._ts)

        self._intent_arrays: IntentArrays = load_intent_arrays(obs["intent"], obs=obs)

        self._proprio = {}
        self._state = {}
        self._fext = {}
        self._grasp_confirmed = {}
        for arm in ("arm_left", "arm_right"):
            g = obs[arm]
            self._proprio[arm] = {
                "q": g["q"][:], "dq": g["dq"][:], "O_T_EE": g["O_T_EE"][:],
                "gripper_width": g["gripper_width"][:],
            }
            self._state[arm] = g["state"][:].astype(int) if "state" in g else np.full(self._T, self.engaged_states[0])
            self._fext[arm] = g["F_ext"][:] if "F_ext" in g else np.zeros((self._T, 6))
            # Real grasp-confirmation signal (ArmControl::updateGraspConfirmation),
            # present on episodes converted after the simulator started logging it,
            # or backfilled offline -- absent on older, non-backfilled episodes.
            if "grasp_confirmed" in g:
                self._grasp_confirmed[arm] = g["grasp_confirmed"][:].astype(bool)

        both = np.ones(self._T, dtype=bool)
        for arm in ("arm_left", "arm_right"):
            both &= np.isin(self._state[arm], self.engaged_states)
        self._engaged = both

        self._img_group = obs["images"] if (self.load_images and "images" in obs) else None
        self._cams = list(self._img_group.keys()) if self._img_group is not None else []

    def _global_features(self, t: int) -> np.ndarray:
        """Assembles the scene-level (non-candidate) feature vector at timestep t:
        the shared intent global features plus per-arm gripper width and wrench,
        which live in the arm telemetry groups rather than the intent log."""
        arr = np.concatenate([
            global_features_at(self._intent_arrays, t),
            self._proprio["arm_left"]["gripper_width"][t:t + 1],
            self._proprio["arm_right"]["gripper_width"][t:t + 1],
            self._fext["arm_left"][t], self._fext["arm_right"][t],
        ]).astype(np.float64)
        assert len(arr) == len(GLOBAL_FEATURE_NAMES), (
            f"global_features length {len(arr)} != GLOBAL_FEATURE_NAMES {len(GLOBAL_FEATURE_NAMES)}; "
            "this composition must match contracts.frame.GLOBAL_FEATURE_NAMES exactly"
        )
        return arr

    def frame_at(self, t: int) -> SensorFrame:
        """Builds the SensorFrame for a single timestep without iterating the
        whole episode; for callers (e.g. playback tools) that need one-off
        frames rather than the full paced stream."""
        return self._frame(t)

    def _frame(self, t: int) -> SensorFrame:
        """Builds the SensorFrame for timestep t."""
        features, mask, types = candidate_features_at(self._intent_arrays, t)
        world_pos = candidate_world_pos_at(self._intent_arrays, t)
        images = {c: self._img_group[c][t] for c in self._cams} if self._img_group is not None else {}
        proprio = {arm: np.concatenate([p["q"][t], p["dq"][t], p["O_T_EE"][t]])
                   for arm, p in self._proprio.items()}
        return SensorFrame(
            timestamp_ns=int(self._ts[t]),
            frame_id=int(self._frame_id[t]),
            candidate_features=features,
            candidate_mask=mask,
            candidate_types=types,
            global_features=self._global_features(t),
            proprio=proprio,
            images=images,
            gaze_valid=bool(self._intent_arrays.gaze_valid[t]),
            engaged=bool(self._engaged[t]),
            grasp_confirmed={arm.removeprefix("arm_"): bool(arr[t])
                              for arm, arr in self._grasp_confirmed.items()},
            candidate_world_pos=world_pos,
        )

    def __len__(self) -> int:
        """Number of grid timesteps in the episode."""
        return self._T

    def __iter__(self) -> Iterator[SensorFrame]:
        """Yields SensorFrames in order, optionally paced to rate_hz for live-like replay."""
        period = 1.0 / self.rate_hz if self.rate_hz else 0.0
        for t in range(self._T):
            tic = time.perf_counter()
            yield self._frame(t)
            if period:
                dt = period - (time.perf_counter() - tic)
                if dt > 0:
                    time.sleep(dt)

    def close(self) -> None:
        """Closes the underlying hdf5 handle."""
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()