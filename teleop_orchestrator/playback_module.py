"""Replays a recorded teleoperation episode's COMMANDS as if an operator were
producing them now: reads the recorded world-frame EE command and gripper
command per arm and emits them as ActionOutput, so run.py's actuator puts them
on exactly the same wire path a policy rollout uses.

Deterministic stand-in for a human at the VR rig -- the point is regression
testing the command path (and everything else the orchestrator runs per tick)
without teleoperating. Pair it with teleop-simulator's
episode_config_server.py --replay for the same scene, and
scripts/compare_rollout_to_demo.py for scoring.

Takes either source:
  - a converted episode.hdf5 (actions/<arm>/O_T_EE_cmd_world), for the archived
    episodes in the store root; both arms share one resampled grid.
  - a raw session folder straight out of the simulator (logs/NNN, holding
    arm_left.csv / arm_right.csv / head.csv), so a session can be replayed
    seconds after it was teleoperated, with no conversion step. Playback needs
    no images, which is the only reason the raw folder is enough. Each arm keeps
    its own logging timeline here rather than being resampled onto a common
    grid, so a raw run and a converted run of the same session are equivalent
    but not bit-identical.

Open loop by construction: commands are resampled onto elapsed wall-clock time,
never onto how far execution actually got. If the arm falls behind, the commands
keep coming -- which is exactly what teleoperation does.

Not to be confused with sources.ReplaySource, which replays an episode's
OBSERVATIONS to drive modules offline and commands nothing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from teleop_orchestrator.contracts import ActionOutput, SensorFrame

if TYPE_CHECKING:  # keeps this module importable without the live UDP stack
    from teleop_orchestrator.live.arm_channel import ArmChannel

ARMS = ("arm_left", "arm_right")
ENGAGED_STATE_DEFAULT = (4,)

_NO_WORLD_FRAME = (
    "{source}: no world-frame EE command found. episode.hdf5 files converted before "
    "episode_to_hdf5.py copied the O_T_EE_cmd_world columns need the world-frame backfill "
    "first; base-frame O_T_EE_cmd cannot be sent on the absolute channel, which expects "
    "world frame (worldAbsoluteToBase).")


def _quat_wxyz(rot: Rotation) -> np.ndarray:
    """scipy (x, y, z, w) -> the (w, x, y, z) ArmCommandMsg/ActionOutput uses."""
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def _pose_arrays(flat16: np.ndarray):
    """Column-major flat 4x4 (libfranka O_T_EE layout) -> positions [T,3], Rotation[T]."""
    m = flat16.reshape(-1, 4, 4).transpose(0, 2, 1)
    return m[:, :3, 3].copy(), Rotation.from_matrix(m[:, :3, :3])


def _csv(path: str):
    """Returns (header, data[N, cols]) for one of the simulator's ';' telemetry CSVs."""
    with open(path) as f:
        header = f.readline().strip().split(";")
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data


def _cols(header, data, prefix: str, count: int):
    """Stacks columns '<prefix>0..count-1' into [N, count]; None if any is absent."""
    names = [f"{prefix}{i}" for i in range(count)]
    if not all(n in header for n in names):
        return None
    return data[:, [header.index(n) for n in names]]


def _read_session(folder: str, replay_head: bool):
    """Raw simulator session folder -> per-arm (ts_ns, cmd16, gripper, state), head."""
    arms = {}
    for arm in ARMS:
        path = os.path.join(folder, f"{arm}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{folder}: no {arm}.csv -- not a simulator session folder")
        header, data = _csv(path)
        cmd = _cols(header, data, "O_T_EE_cmd_world_", 16)
        if cmd is None:
            raise KeyError(_NO_WORLD_FRAME.format(source=path))
        arms[arm] = (data[:, header.index("wall_clock_ns")].astype(np.int64), cmd,
                     data[:, header.index("gripper_cmd")].astype(float),
                     data[:, header.index("state")].astype(int))

    head = None
    head_path = os.path.join(folder, "head.csv")
    if replay_head and os.path.exists(head_path):
        header, data = _csv(head_path)
        q_cmd = _cols(header, data, "q_cmd_", 2)
        if q_cmd is not None:
            head = (data[:, header.index("wall_clock_ns")].astype(np.int64), q_cmd)
    return arms, head


def _read_episode(path: str, replay_head: bool):
    """Converted episode.hdf5 -> the same shape as _read_session, on a shared grid."""
    arms = {}
    with h5py.File(path, "r") as f:
        ts = f["observations/timestamp_ns"][:].astype(np.int64)
        for arm in ARMS:
            cmd_key = f"actions/{arm}/O_T_EE_cmd_world"
            if cmd_key not in f:
                raise KeyError(_NO_WORLD_FRAME.format(source=f"{path} ({cmd_key})"))
            state_key = f"observations/{arm}/state"
            state = (f[state_key][:].astype(int) if state_key in f
                     else np.full(len(ts), ENGAGED_STATE_DEFAULT[0]))
            arms[arm] = (ts, f[cmd_key][:], f[f"actions/{arm}/gripper_cmd"][:].astype(float), state)
        head_key = "observations/head/q_cmd"
        head = (ts, f[head_key][:].astype(float)) if replay_head and head_key in f else None
    return arms, head


class PlaybackModule:
    """Emits one recorded episode's command stream, resampled onto the live tick
    clock. reset()/step() shape matches contracts.Module, and the output is the
    same ActionOutput a policy would produce, so nothing downstream needs to
    know which of the two is driving.

    arm channels are optional and used only for start_ramp_s (blending out of
    wherever the arms actually are into the episode's first command, so playback
    never opens with a step input to two 7-DoF arms). Without them, or before
    any ArmStateMsg has arrived, the first command is issued as-is.
    """

    def __init__(self, episode_path: str,
                 arm_left: Optional["ArmChannel"] = None,
                 arm_right: Optional["ArmChannel"] = None,
                 speed: float = 1.0, start_ramp_s: float = 2.0,
                 engaged_states=ENGAGED_STATE_DEFAULT, replay_head: bool = True):
        self.path = episode_path
        self.speed = float(speed)
        self.start_ramp_s = max(float(start_ramp_s), 0.0)
        self._channels = {"arm_left": arm_left, "arm_right": arm_right}
        self._t0_ns: Optional[int] = None
        self._ramp: dict = {}
        self.finished = False

        self.source_kind = "session" if os.path.isdir(episode_path) else "episode"
        reader = _read_session if self.source_kind == "session" else _read_episode
        arms, head = reader(episode_path, replay_head)
        self._assemble(arms, head, tuple(engaged_states))

        print(f"[PlaybackModule] {episode_path} ({self.source_kind}): {self.duration_s:.1f}s engaged, "
              f"{min(len(t) for t in self._t.values())}-{max(len(t) for t in self._t.values())} samples "
              f"per arm, speed {self.speed}{'' if self._head is not None else ', no head track'}")

    def _assemble(self, arms: dict, head, engaged_states: tuple) -> None:
        """Clips every stream to the window where BOTH arms were engaged, re-zeroes
        the clock on it, and precomputes the interpolators. Streams keep their own
        sample times: one grid when they came from an episode.hdf5, each arm's own
        logging rate when they came from raw CSVs."""
        windows = []
        for arm, (ts, _, _, state) in arms.items():
            idx = np.flatnonzero(np.isin(state, engaged_states))
            if len(idx) < 2:
                raise ValueError(f"{self.path}: {arm} has fewer than two engaged samples")
            windows.append((ts[idx[0]], ts[idx[-1]]))
        t0, t1 = max(w[0] for w in windows), min(w[1] for w in windows)
        if t1 <= t0:
            raise ValueError(f"{self.path}: the arms are never engaged at the same time")
        self.duration_s = float((t1 - t0) * 1e-9)

        self._t, self._pos, self._slerp, self._grip = {}, {}, {}, {}
        for arm, (ts, cmd, grip, _) in arms.items():
            keep = self._clip(ts, t0, t1, arm)
            self._t[arm] = (ts[keep] - t0) * 1e-9
            pos, rot = _pose_arrays(cmd[keep])
            self._pos[arm] = pos
            self._slerp[arm] = Slerp(self._t[arm], rot)
            self._grip[arm] = grip[keep]

        self._head_t = self._head = None
        if head is not None:
            ts, q_cmd = head
            keep = self._clip(ts, t0, t1, "head")
            self._head_t = (ts[keep] - t0) * 1e-9
            # q_cmd_0/q_cmd_1 are the head's absolute pan/tilt joint targets
            # (head_control.cpp: cmd.pan -> q_target(0), cmd.tilt -> q_target(1)),
            # so they replay directly. Using them puts head_cam_left where it was
            # during the demo instead of at run.py's fixed look-down.
            self._head = q_cmd[keep]

    def _clip(self, ts: np.ndarray, t0: int, t1: int, label: str) -> np.ndarray:
        """Indices of the samples inside [t0, t1], with duplicate timestamps
        dropped -- Slerp needs a strictly increasing time base and a real-time
        logger can emit two rows in the same nanosecond."""
        idx = np.flatnonzero((ts >= t0) & (ts <= t1))
        if len(idx) < 2:
            raise ValueError(f"{self.path}: {label} has fewer than two samples in the engaged window")
        return idx[np.concatenate([[0], np.flatnonzero(np.diff(ts[idx]) > 0) + 1])]

    def reset(self) -> None:
        """Re-anchors the playback clock and drops the ramp captured last run."""
        self._t0_ns = None
        self._ramp = {}
        self.finished = False

    @property
    def name(self) -> str:
        return "PlaybackModule"

    def step(self, frame: SensorFrame) -> ActionOutput:
        if self._t0_ns is None:
            self._t0_ns = frame.timestamp_ns
            self._capture_ramp()

        elapsed = (frame.timestamp_ns - self._t0_ns) * 1e-9
        t = (elapsed - self.start_ramp_s) * self.speed
        if t > self.duration_s:
            self.finished = True
            return ActionOutput(ee_pose={}, gripper={}, extras={"finished": "episode exhausted"})

        ee_pose, gripper = {}, {}
        for arm in ARMS:
            pos, quat = self._ramped_pose(arm, elapsed) if t < 0.0 else self._pose_at(arm, t)
            ee_pose[arm] = np.concatenate([pos, quat])
            # Zero-order hold, not interpolation: gripper_cmd is a binary
            # 0.0/0.08 width and an interpolated value would invent transitions
            # the operator never commanded.
            gripper[arm] = float(self._grip[arm][self._hold_index(arm, max(t, 0.0))])

        extras = {"playback_t_s": max(t, 0.0), "playback_progress": max(t, 0.0) / self.duration_s}
        if self._head is not None:
            extras["head_pan"] = float(np.interp(max(t, 0.0), self._head_t, self._head[:, 0]))
            extras["head_tilt"] = float(np.interp(max(t, 0.0), self._head_t, self._head[:, 1]))
        return ActionOutput(ee_pose=ee_pose, gripper=gripper, extras=extras)

    def _hold_index(self, arm: str, t: float) -> int:
        """Index of that arm's last recorded sample at or before t."""
        return max(int(np.searchsorted(self._t[arm], t, side="right")) - 1, 0)

    def _pose_at(self, arm: str, t: float):
        """Recorded command at t: linear in position, slerp in orientation. t is
        clamped to this arm's own sample range, which can end microseconds inside
        the common window when each arm logs on its own clock."""
        ts = self._t[arm]
        pos = np.array([np.interp(t, ts, self._pos[arm][:, i]) for i in range(3)])
        return pos, _quat_wxyz(self._slerp[arm](min(max(t, ts[0]), ts[-1])))

    def _capture_ramp(self) -> None:
        """Snapshots each arm's measured pose at t0 and plans the blend into the
        episode's first command. None means no ramp for that arm."""
        for arm in ARMS:
            channel = self._channels.get(arm)
            state = channel.latest() if channel is not None else None
            if state is None or self.start_ramp_s == 0.0:
                self._ramp[arm] = None
                continue
            w, x, y, z = state.quaternion
            start = Rotation.from_quat([[x, y, z, w]])
            target = self._slerp[arm](self._t[arm][0])
            self._ramp[arm] = (np.asarray(state.position, dtype=float), self._pos[arm][0],
                               Slerp([0.0, 1.0], Rotation.concatenate([start, target])))

    def _ramped_pose(self, arm: str, elapsed: float):
        ramp = self._ramp.get(arm)
        if ramp is None:
            return self._pos[arm][0], _quat_wxyz(self._slerp[arm](self._t[arm][0]))
        p0, p1, slerp = ramp
        a = min(elapsed / self.start_ramp_s, 1.0)
        return (1.0 - a) * p0 + a * p1, _quat_wxyz(slerp(a))
