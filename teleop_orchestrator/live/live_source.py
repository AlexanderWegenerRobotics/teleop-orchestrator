"""Live counterpart to sources.py's ReplaySource: composites an ObjectSource
(candidate/bin geometry), two ArmChannels, a HeadChannel, and a GazeReceiver
into one SensorFrame per tick, reusing the exact CANDIDATE_FEATURE_NAMES/
GLOBAL_FEATURE_NAMES composition contracts/features.py and ReplaySource
already establish, so live and offline paths can't drift on what a feature
means.

Tick anchor: the ObjectSource's frame_id (SceneObjectsMsg, published once
per Avatar tick -- see camera_geometry.py's module docstring for the
frame_id space this shares with gaze). Each new frame_id becomes one
SensorFrame: gaze joins by exact frame_id (GazeReceiver.get), arm/head state
join by "most recently received" (ArmStateMsg/HeadStateMsg carry no
frame_id -- see wire.py's MsgHeader).

Known gaps, flagged rather than silently approximated:
  - proprio is built to the same 30-element q(7)+dq(7)+O_T_EE(16) layout
    ReplaySource uses, but only indices [26:29] (O_T_EE's translation) are
    real live data -- verified by grepping every model family in
    teleop-intent, the ONLY place any of them reads SensorFrame.proprio
    directly is models/hmm/features.py's _EE_POS_SLICE = slice(26, 29),
    which is exactly ArmStateMsg.position (both are the arm's base-frame EE
    position; HMM's phase_features does its own causal finite-difference for
    velocity, so dq was never actually needed). q/dq and O_T_EE's rotation
    are zeroed -- a future consumer reading anything outside [26:29] would
    get zeros, not real data; check here before relying on it.
  - PolicyModule derives its own pos+rot6d proprio composition directly from
    ArmChannel (see teleop-policy/dataset/transforms.py's convention), not
    through this field, since ACT's proprio shape is different again.
  - global_features' fext_left/right_* come from ArmStateMsg.F_ext (the
    avatar's O_F_ext_hat_K estimate, base frame) since 2026-10-13; an older
    avatar is rejected on packet size, so they are never silently zero.
  - global_features' ee_belief_left/right are zeroed: that was the C++
    belief filter's own gaze-on-EE posterior, which no longer runs live now
    that gaze fusion moved to the Python models. contracts/features.py
    already treats an absent ee_belief as zeros (global_features_at), so
    this is the existing supported fallback, not new behavior -- but if a
    currently-deployed model was trained expecting nonzero ee_belief, this
    is a live distribution shift worth checking.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from teleop_orchestrator.contracts import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, SensorFrame
from teleop_orchestrator.contracts.features import (CAND_UNKNOWN, NOT_PROJECTED, append_goal_candidates,
                                                    project_static)

from . import wire
from .arm_channel import ArmChannel
from .camera_geometry import CameraParams, project_to_ray
from .gaze_receiver import GazeReceiver
from .head_channel import HeadChannel
from .object_source import ObjectSource
from .shared_frame_reader import SharedFrameReader



class LiveSource:
    """Live analog of ReplaySource: blocks (briefly polling) for the next new
    object-geometry tick, then assembles one SensorFrame from whatever the
    other channels most recently have. Same iterable-of-SensorFrame contract
    as ReplaySource, so Orchestrator.run() doesn't know the difference."""

    def __init__(self, *, object_source: ObjectSource, arm_left: ArmChannel, arm_right: ArmChannel,
                 head: HeadChannel, gaze: GazeReceiver, cameras: dict[str, SharedFrameReader],
                 head_position, camera_position, camera: Optional[CameraParams] = None,
                 n_candidates: int = 8, poll_interval_s: float = 0.005, tick_timeout_s: float = 2.0,
                 twin_left: Optional[ArmChannel] = None, twin_right: Optional[ArmChannel] = None,
                 tick_on: str = "frame", wait_forever: bool = False, static_camera: bool = False, goals: Optional[dict] = None):
        # camera: camera_params.json entry (intrinsics for gaze, T_world_cam for a static camera), re-read when the
        # sim rewrites it. static_camera: candidates are projected through camera.T_world_cam, like the avatar's
        # static intention mode and the logged desk episodes, instead of through the head chain.
        # Gaze pixels are scaled by the texture size each gaze sample reports (CameraParams.gaze_scale).
        # goals: name -> world position of the task's board slots, appended as CAND_GOAL candidates.
        # Gaze is tagged with the video frame the operator saw, which lags the newest scene frame by the
        # video latency, so an exact frame match never exists at tick time. The newest sample (bounded by
        # GazeReceiver.max_staleness_s) is used instead; with a static camera the projection is the same.
        self._objects = object_source
        self._arm_left = arm_left
        self._arm_right = arm_right
        self._twin = {"arm_left": twin_left, "arm_right": twin_right}
        self._head = head
        self._gaze = gaze
        self._cameras = cameras
        self._head_position = head_position
        self._camera_position = camera_position
        self._camera = camera
        self._n_candidates = n_candidates
        self._static_camera = static_camera
        if static_camera and (camera is None or camera.T_world_cam is None):
            raise ValueError("static_camera needs camera_params.json with a T_world_cam for the camera")
        self._goal_names = list(goals) if goals else []
        self._goal_pos = (np.asarray([goals[g] for g in self._goal_names], dtype=np.float64).reshape(-1, 3)
                          if goals else np.zeros((0, 3)))
        self._poll_interval_s = poll_interval_s
        self._tick_timeout_s = tick_timeout_s
        self._last_frame_id: Optional[int] = None
        self._tick_on = tick_on
        self._wait_forever = wait_forever
        self._was_live = False
        self._last_tick_id: Optional[int] = None

    def reset(self) -> None:
        """Clears buffered state on the live inputs that carry it, at an episode boundary."""
        self._objects.reset()
        self._gaze.reset()
        self._last_frame_id = None
        self._last_tick_id = None

    def _wait_for_next_tick(self):
        """Blocks (polling) until a new ObjectFrame arrives, or returns None on timeout."""
        deadline = time.monotonic() + self._tick_timeout_s
        while self._wait_forever or time.monotonic() < deadline:
            frame = self._objects.latest()
            if frame is not None and (frame.tick_id != self._last_tick_id if self._tick_on == "tick" else frame.frame_id != self._last_frame_id):
                if not self._was_live:
                    print("[LiveSource] avatar feed up")
                    self._was_live = True
                return frame
            if self._was_live and time.monotonic() >= deadline:
                print("[LiveSource] avatar feed lost, idling until it returns")
                self._was_live = False
                self._last_tick_id = None
                self._last_frame_id = None
            time.sleep(self._poll_interval_s)
        return None

    def _rays(self, points: np.ndarray, head_pan: float, head_tilt: float) -> np.ndarray:
        """Normalized ray coords [N, 2] for world points, through the static camera if set, else the head chain."""
        if len(points) == 0:
            return np.zeros((0, 2))
        if self._static_camera:
            return project_static(points, self._camera.T_world_cam)
        out = np.full((len(points), 2), NOT_PROJECTED)
        for i, p in enumerate(points):
            ray = project_to_ray(p, head_pan, head_tilt, self._head_position, self._camera_position)
            if ray is not None:
                out[i] = ray
        return out

    def _candidate_arrays(self, obj_frame, head_pan: float, head_tilt: float, ee_left_pos, ee_right_pos):
        """Builds (features, mask, types, world_pos, names) padded to n_candidates, with the goals appended."""
        n = min(len(obj_frame.slots), self._n_candidates)
        features = np.zeros((self._n_candidates, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float64)
        mask = np.zeros(self._n_candidates, dtype=bool)
        types = np.full(self._n_candidates, CAND_UNKNOWN, dtype=np.int64)
        world_pos = np.full((self._n_candidates, 3), np.nan, dtype=np.float64)
        rays = self._rays(np.asarray([s.position for s in obj_frame.slots[:n]], dtype=np.float64).reshape(-1, 3),
                          head_pan, head_tilt)
        for i in range(n):
            slot = obj_frame.slots[i]
            px_u, px_v = rays[i]
            dist_left = float(np.linalg.norm(np.asarray(slot.position) - np.asarray(ee_left_pos)))
            dist_right = float(np.linalg.norm(np.asarray(slot.position) - np.asarray(ee_right_pos)))
            features[i] = (px_u, px_v, dist_left, dist_right)
            mask[i] = True
            types[i] = slot.type
            world_pos[i] = slot.position
        names = [s.name for s in obj_frame.slots[:n]]
        if self._goal_names:
            return append_goal_candidates(features, mask, types, world_pos, names, self._goal_names, self._goal_pos,
                                          self._rays(self._goal_pos, head_pan, head_tilt), ee_left_pos, ee_right_pos)
        return features, mask, types, world_pos, names + [""] * (self._n_candidates - n)

    def _global_features(self, gaze_ray, gaze_valid: bool, left, right) -> np.ndarray:
        """Assembles GLOBAL_FEATURE_NAMES's values from live channels -- see
        module docstring for the fext/ee_belief gaps."""
        gaze_x, gaze_y = gaze_ray if gaze_ray is not None else (0.0, 0.0)
        gripper_left = left.gripper_width if left else 0.0
        gripper_right = right.gripper_width if right else 0.0
        ee_left = left.position if left else (0.0, 0.0, 0.0)
        ee_right = right.position if right else (0.0, 0.0, 0.0)
        arr = np.concatenate([
            [gaze_x, gaze_y, float(gaze_valid)],
            [gripper_left, gripper_right],           # "gripper_left/right" -- same live reading as gripper_width below
            list(ee_left), list(ee_right),
            [0.0, 0.0],                                # ee_belief_left/right -- see module docstring
            [gripper_left, gripper_right],            # gripper_width_left/right
            list(left.F_ext) if left else [0.0] * 6, list(right.F_ext) if right else [0.0] * 6,
        ]).astype(np.float64)
        assert len(arr) == len(GLOBAL_FEATURE_NAMES), (
            f"global_features length {len(arr)} != GLOBAL_FEATURE_NAMES {len(GLOBAL_FEATURE_NAMES)}; "
            "this composition must match contracts.frame.GLOBAL_FEATURE_NAMES exactly"
        )
        return arr

    @staticmethod
    def _build_proprio(arm_state) -> np.ndarray:
        """Builds the 30-element q(7)+dq(7)+O_T_EE(16) array ReplaySource's
        shape expects, real data only at [26:29] (O_T_EE's translation) --
        see module docstring for why that's the only slice that matters."""
        arr = np.zeros(30, dtype=np.float64)
        arr[26:29] = arm_state.position
        return arr

    def __iter__(self) -> Iterator[SensorFrame]:
        while True:
            obj_frame = self._wait_for_next_tick()
            if obj_frame is None:
                return  # sim went quiet -- end the stream rather than spin forever
            self._last_frame_id = obj_frame.frame_id
            self._last_tick_id = obj_frame.tick_id

            left = self._arm_left.latest()
            right = self._arm_right.latest()
            # Head pose: the scene publish first, the head channel only as a
            # fallback. The avatar's head transmission is point-to-point like
            # every other outbound channel, so when the VR interface is also
            # running it owns that port and self._head never receives anything.
            # The same pan/tilt rides SceneObjectsMsg, on a socket with no such
            # contention, which is what this reads.
            head = self._head.latest()
            if head is not None:
                head_pan, head_tilt = head.pan, head.tilt
            else:
                head_pan, head_tilt = obj_frame.head_pan, obj_frame.head_tilt

            if self._camera is not None:
                self._camera.refresh()
            gaze_sample = self._gaze.get(obj_frame.frame_id) or self._gaze.latest()
            gaze_ray = None
            if gaze_sample is not None and self._camera is not None:
                gaze_ray = self._camera.gaze_ray(gaze_sample.gaze_px_x, gaze_sample.gaze_px_y,
                                                 gaze_sample.image_w, gaze_sample.image_h)

            ee_left_pos = left.position if left else (0.0, 0.0, 0.0)
            ee_right_pos = right.position if right else (0.0, 0.0, 0.0)
            cand_features, cand_mask, cand_types, world_pos, cand_names = self._candidate_arrays(
                obj_frame, head_pan, head_tilt, ee_left_pos, ee_right_pos)

            images = {name: frame for name, reader in self._cameras.items()
                      if (frame := reader.read()) is not None}

            grasp_confirmed = {}
            if left is not None:
                grasp_confirmed["left"] = left.grasp_state == wire.GraspState.HELD
            if right is not None:
                grasp_confirmed["right"] = right.grasp_state == wire.GraspState.HELD

            proprio = {name: (self._build_proprio(st) if st is not None else np.zeros(30, dtype=np.float64))
                       for name, st in (("arm_left", left), ("arm_right", right))}

            present = [st for st in (left, right) if st is not None]
            if obj_frame.state != 255:
                engaged = obj_frame.state == wire.SysState.ENGAGED
            else:
                engaged = bool(present) and all(st.state == wire.SysState.ENGAGED for st in present)

            arm_states = {"remote": {"arm_left": left, "arm_right": right}}
            if any(self._twin.values()):
                arm_states["twin"] = {k: (c.latest() if c else None) for k, c in self._twin.items()}

            yield SensorFrame(
                timestamp_ns=obj_frame.timestamp_ns,
                frame_id=obj_frame.frame_id,
                candidate_features=cand_features,
                candidate_mask=cand_mask,
                candidate_types=cand_types,
                global_features=self._global_features(gaze_ray, gaze_ray is not None, left, right),
                proprio=proprio,
                images=images,
                gaze_valid=gaze_ray is not None,
                engaged=engaged,
                authority=obj_frame.authority,
                # Resolved above (head channel first, scene publish as
                # fallback). Passed on so no consumer has to repeat it.
                head_pan=head_pan,
                head_tilt=head_tilt,
                grasp_confirmed=grasp_confirmed,
                candidate_world_pos=world_pos,
                arm_states=arm_states,
                candidate_names=cand_names,
                operator=self._gaze.operator(),
                ee_world_pos={side: np.asarray(st.position, dtype=float)
                              for side, st in (('left', left), ('right', right)) if st is not None},
            )
