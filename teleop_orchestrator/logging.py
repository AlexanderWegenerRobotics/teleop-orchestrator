"""Records what happened during a run (frame inputs + module outputs) to a run.hdf5."""

from __future__ import annotations

import os
import time
from collections import defaultdict

import h5py
import numpy as np

from .contracts import SensorFrame, IntentOutput, ActionOutput, AssistOutput


class RunLogger:
    """Buffers per-tick frame meta and module outputs, then writes a self-contained run.hdf5.

    Provider-agnostic: an offline replay and a live run produce the same format,
    the live one simply lacking ground-truth labels. Scoring is a separate
    offline step that reads this log plus labels — it does not live here.
    """

    def __init__(self):
        self._frame = defaultdict(list)     # frame-level fields
        self._mod = defaultdict(list)       # "module/.../field" -> list of arrays
        self._modules: list[str] = []
        # What was actually put on the wire, as opposed to what the module
        # predicted. These diverge whenever an adapter sits between the two --
        # e.g. the gripper, where the policy emits a width in metres and
        # ArmCommandMsg carries a boolean close flag (see run.py). Logging only
        # the module output hid a wiring bug that made the gripper physically
        # unable to close; keep both so the two are always comparable.
        self._cmd = defaultdict(list)

    @property
    def n_ticks(self) -> int:
        """Ticks recorded so far; zero means the source never produced a frame."""
        return len(self._frame.get("timestamp_ns", []))

    def log_gripper(self, side: str, width: float, close_flag: float) -> None:
        """Records the commanded gripper for one arm this tick: the policy's
        raw predicted width and the boolean flag actually transmitted."""
        self._cmd[f"{side}/gripper_width_pred"].append(float(width))
        self._cmd[f"{side}/gripper_close_flag"].append(float(close_flag))

    def log_arm_state(self, side: str, state) -> None:
        """Records the MEASURED arm state this tick (ArmStateMsg, see wire.py).

        Without this the log only ever contained what the policy wanted; there
        was no way to tell a policy that never asked to grasp from a command
        path that dropped the request. gripper_width is the real finger
        opening, grasp_state is arm_control.cpp's sensor-derived confirmation
        (updateGraspConfirmation), so "asked to close / fingers moved / holding
        something" become three separately checkable things.

        state may be None when the feed is stale (ArmChannel.latest() returns
        None rather than a stale pose); NaN keeps the series aligned with the
        tick index instead of silently shortening it.
        """
        if state is None:
            self._cmd[f"{side}/meas_gripper_width"].append(float("nan"))
            self._cmd[f"{side}/meas_grasp_state"].append(float("nan"))
            self._cmd[f"{side}/meas_ee_pos"].append([float("nan")] * 3)
            return
        self._cmd[f"{side}/meas_gripper_width"].append(float(state.gripper_width))
        self._cmd[f"{side}/meas_grasp_state"].append(float(state.grasp_state))
        self._cmd[f"{side}/meas_ee_pos"].append([float(v) for v in state.position])

    def record(self, frame: SensorFrame, outputs: dict) -> None:
        """Appends one tick: frame metadata and each module's serialized output."""
        self._frame["timestamp_ns"].append(frame.timestamp_ns)
        self._frame["frame_id"].append(frame.frame_id)
        self._frame["engaged"].append(frame.engaged)
        self._frame["gaze_valid"].append(frame.gaze_valid)
        self._frame["usable"].append(frame.usable)
        # Who held each arm this tick, as the avatar reported it. One series per
        # arm rather than one for the robot: the two genuinely differ during an
        # intervention, and a single number could only be right about one.
        #
        # This is also the join between this log and the avatar's arm.csv. The
        # same quantity is recorded on both sides of the link, produced
        # independently, so the two disagreeing is visible rather than silent.
        for side in ("arm_left", "arm_right"):
            self._frame[f"authority_{side}"].append(int((frame.authority or {}).get(side, 255)))
        self._frame["candidate_types"].append(frame.candidate_types)
        self._frame["candidate_mask"].append(frame.candidate_mask)
        # Where the candidates actually were, not just how many there were.
        # candidate_features columns are CANDIDATE_FEATURE_NAMES
        # (px_u, px_v, dist_left, dist_right), so this answers both "was the
        # gripper near a parcel when it tried to grasp" and "was the parcel
        # even framed where the model expects" -- neither of which the
        # types/mask pair could tell you. candidate_world_pos is
        # sim-privileged ground truth and is None on sources that don't have
        # it, so it is only recorded when present (a run either has the
        # dataset throughout or never).
        self._frame["candidate_features"].append(frame.candidate_features)
        if frame.candidate_world_pos is not None:
            self._frame["candidate_world_pos"].append(frame.candidate_world_pos)
        for name, out in outputs.items():
            if name not in self._modules:
                self._modules.append(name)
            _serialize(name, out, self._mod)

    def save(self, path: str, meta: dict | None = None) -> None:
        """Writes all buffered records to run.hdf5 under frames/ and modules/."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.attrs["modules"] = np.array(self._modules, dtype=h5py.string_dtype())
            for k, v in (meta or {}).items():
                f.attrs[k] = v
            fg = f.create_group("frames")
            n = len(self._frame.get("timestamp_ns", []))
            for k, v in self._frame.items():
                # candidate_world_pos is skipped on ticks where it was absent,
                # so a partially-populated series would silently misalign with
                # the tick index. Drop it rather than write something that
                # looks aligned and isn't.
                if len(v) != n:
                    print(f"[RunLogger] skipping frames/{k}: {len(v)} of {n} ticks populated")
                    continue
                fg.create_dataset(k, data=np.asarray(v))
            mg = f.create_group("modules")
            for path_key, seq in self._mod.items():
                mg.create_dataset(path_key, data=np.asarray(seq))
            if self._cmd:
                cg = f.create_group("commanded")
                for path_key, seq in self._cmd.items():
                    cg.create_dataset(path_key, data=np.asarray(seq))


def _serialize(name: str, out, store: dict) -> None:
    """Dispatches a module output to per-field buffers keyed by dotted path."""
    if isinstance(out, IntentOutput):
        for arm in ("left", "right"):
            ai = out.arm(arm)
            store[f"{name}/{arm}/phase_posterior"].append(ai.phase_posterior)
            store[f"{name}/{arm}/target_posterior"].append(ai.target_posterior)
            if ai.location_posterior is not None:
                store[f"{name}/{arm}/location_posterior"].append(ai.location_posterior)
    elif isinstance(out, ActionOutput):
        for arm, pose in out.ee_pose.items():
            store[f"{name}/{arm}/ee_pose"].append(pose)
            store[f"{name}/{arm}/gripper"].append(out.gripper.get(arm, 0.0))
        # Scalar extras (e.g. PolicyModule's raw pre-ensemble chunk endpoints)
        # ride alongside as their own series. Only floats: extras is also used
        # for string diagnostics like {"skipped": ...}, which have no place in
        # a fixed-width numeric dataset.
        for key, val in (out.extras or {}).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                store[f"{name}/extras/{key}"].append(float(val))
    elif isinstance(out, AssistOutput):
        store[f"{name}/active"].append(float(out.active))
        store[f"{name}/target"].append(out.target)
    # unknown output types are skipped; add a handler when a new one appears