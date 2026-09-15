"""PlaybackModule: resampling, gripper hold, ramp-in, and end-of-episode."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_orchestrator.contracts import SensorFrame
from teleop_orchestrator.playback_module import PlaybackModule

RATE_HZ = 30.0
N = 90  # 3 s


def _flat16(pos, rot):
    m = np.zeros((len(pos), 4, 4))
    m[:, :3, :3] = rot.as_matrix()
    m[:, :3, 3] = pos
    m[:, 3, 3] = 1.0
    return m.transpose(0, 2, 1).reshape(-1, 16)  # column-major, as libfranka logs it


@pytest.fixture
def episode(tmp_path):
    """A 3 s episode: x ramps 0 -> 1 m, yaw 0 -> 90 deg, gripper closes at 1.5 s.
    First and last 10 ticks are not ENGAGED, so the engaged-window slice is tested too."""
    import h5py

    path = tmp_path / "episode.hdf5"
    t = np.arange(N) / RATE_HZ
    pos = np.stack([t / t[-1], np.zeros(N), np.full(N, 0.5)], axis=1)
    rot = Rotation.from_euler("z", np.linspace(0, np.pi / 2, N)[:, None])
    grip = np.where(t < 1.5, 0.08, 0.0)
    state = np.full(N, 4)
    state[:10] = 1
    state[-10:] = 1

    with h5py.File(path, "w") as f:
        f.create_dataset("observations/timestamp_ns", data=(t * 1e9).astype(np.int64))
        f.create_dataset("observations/head/q_cmd", data=np.stack([t * 0.1, -t * 0.2], axis=1))
        for arm in ("arm_left", "arm_right"):
            f.create_dataset(f"observations/{arm}/state", data=state)
            f.create_dataset(f"actions/{arm}/O_T_EE_cmd_world", data=_flat16(pos, rot))
            f.create_dataset(f"actions/{arm}/gripper_cmd", data=grip)
    return str(path), t, pos, rot


@pytest.fixture
def session(tmp_path, episode):
    """The same trajectory as a raw simulator session folder, with the two arms on
    deliberately different logging clocks (90 Hz left, 60 Hz right)."""
    _, t, pos, rot = episode
    folder = tmp_path / "021"
    folder.mkdir()

    def write_arm(name, hz):
        ts = np.arange(0.0, t[-1], 1.0 / hz)
        p = np.stack([np.interp(ts, t, pos[:, i]) for i in range(3)], axis=1)
        r = Rotation.from_euler("z", np.interp(ts, t, rot.as_euler("xyz")[:, 2])[:, None])
        cmd = _flat16(p, r)
        state = np.where((ts >= t[10]) & (ts <= t[-11]), 4, 1)
        grip = np.where(ts < 1.5, 0.08, 0.0)
        header = (["time", "wall_clock_ns"] + [f"O_T_EE_cmd_world_{i}" for i in range(16)]
                  + ["gripper_cmd", "state"])
        rows = np.column_stack([ts, ts * 1e9, cmd, grip, state])
        with open(folder / name, "w") as f:
            f.write(";".join(header) + "\n")
            for row in rows:
                f.write(";".join(f"{v:.9f}" for v in row) + "\n")

    write_arm("arm_left.csv", 90.0)
    write_arm("arm_right.csv", 60.0)
    with open(folder / "head.csv", "w") as f:
        f.write("time;wall_clock_ns;q_cmd_0;q_cmd_1;state\n")
        for ts in np.arange(0.0, t[-1], 1 / 50.0):
            f.write(f"{ts};{ts * 1e9:.0f};{ts * 0.1};{-ts * 0.2};4\n")
    return str(folder), pos


def _frame(t_s: float) -> SensorFrame:
    return SensorFrame(timestamp_ns=int(t_s * 1e9), frame_id=int(t_s * RATE_HZ),
                       candidate_features=np.zeros((1, 4)), candidate_mask=np.ones(1, bool),
                       candidate_types=np.zeros(1, int), global_features=np.zeros(27))


def test_engaged_window_only(episode):
    path, t, _, _ = episode
    pb = PlaybackModule(path, start_ramp_s=0.0)
    # 90 ticks minus 10 leading and 10 trailing non-engaged ones, at 30 Hz
    assert pb.duration_s == pytest.approx((N - 20 - 1) / RATE_HZ)


def test_interpolates_between_recorded_samples(episode):
    path, t, pos, rot = episode
    pb = PlaybackModule(path, start_ramp_s=0.0)
    pb.reset()
    pb.step(_frame(0.0))
    out = pb.step(_frame(0.5 / RATE_HZ))  # half a sample into the engaged window

    expected_pos = 0.5 * (pos[10] + pos[11])
    assert out.ee_pose["arm_left"][:3] == pytest.approx(expected_pos, abs=1e-9)

    w, x, y, z = out.ee_pose["arm_left"][3:]
    got = Rotation.from_quat([x, y, z, w])
    mid = rot[10].as_quat() @ got.as_quat()  # slerp midpoint sits between the two samples
    assert abs(mid) > abs(rot[10].as_quat() @ rot[11].as_quat())


def test_gripper_is_held_not_interpolated(episode):
    path, _, _, _ = episode
    pb = PlaybackModule(path, start_ramp_s=0.0)
    pb.reset()
    pb.step(_frame(0.0))
    for t_s in np.arange(0.0, pb.duration_s, 0.01):
        assert pb.step(_frame(t_s)).gripper["arm_right"] in (0.0, 0.08)


def test_head_track_rides_in_extras(episode):
    path, _, _, _ = episode
    pb = PlaybackModule(path, start_ramp_s=0.0)
    pb.reset()
    pb.step(_frame(0.0))
    extras = pb.step(_frame(1.0)).extras
    assert extras["head_pan"] > 0 and extras["head_tilt"] < 0

    pb_no_head = PlaybackModule(path, start_ramp_s=0.0, replay_head=False)
    pb_no_head.reset()
    assert "head_pan" not in pb_no_head.step(_frame(0.0)).extras


def test_ramp_starts_at_the_measured_pose(episode):
    path, _, pos, _ = episode

    class FakeChannel:
        def latest(self):
            return type("S", (), {"position": (0.0, 0.0, 1.0), "quaternion": (1.0, 0.0, 0.0, 0.0)})()

    pb = PlaybackModule(path, FakeChannel(), FakeChannel(), start_ramp_s=2.0)
    pb.reset()
    first = pb.step(_frame(0.0)).ee_pose["arm_left"][:3]
    assert first == pytest.approx([0.0, 0.0, 1.0])

    mid = pb.step(_frame(1.0)).ee_pose["arm_left"][:3]
    assert mid[2] == pytest.approx(0.5 * (1.0 + pos[10][2]))

    after = pb.step(_frame(2.0)).ee_pose["arm_left"][:3]
    assert after == pytest.approx(pos[10], abs=1e-9)
    assert not pb.finished


def test_finishes_and_stops_commanding(episode):
    path, _, _, _ = episode
    pb = PlaybackModule(path, start_ramp_s=0.0)
    pb.reset()
    pb.step(_frame(0.0))
    assert not pb.finished
    out = pb.step(_frame(pb.duration_s + 0.1))
    assert pb.finished and out.ee_pose == {}

    pb.reset()
    assert not pb.finished


def test_speed_scales_the_clock(episode):
    path, _, _, _ = episode
    pb = PlaybackModule(path, start_ramp_s=0.0, speed=2.0)
    pb.reset()
    pb.step(_frame(0.0))
    assert pb.step(_frame(1.0)).extras["playback_t_s"] == pytest.approx(2.0)


def test_session_folder_matches_the_converted_episode(session, episode):
    """A raw session folder plays back the same trajectory as the converted file,
    despite each arm running on its own logging clock."""
    folder, pos = session
    path, _, _, _ = episode

    raw = PlaybackModule(folder, start_ramp_s=0.0)
    converted = PlaybackModule(path, start_ramp_s=0.0)
    assert raw.source_kind == "session" and converted.source_kind == "episode"
    assert raw.duration_s == pytest.approx(converted.duration_s, abs=0.05)

    raw.reset(), converted.reset()
    raw.step(_frame(0.0)), converted.step(_frame(0.0))
    for t_s in (0.3, 1.0, 1.7):
        a = raw.step(_frame(t_s)).ee_pose["arm_right"]
        b = converted.step(_frame(t_s)).ee_pose["arm_right"]
        assert a[:3] == pytest.approx(b[:3], abs=2e-3)
        assert raw.step(_frame(t_s)).gripper["arm_left"] == converted.step(_frame(t_s)).gripper["arm_left"]


def test_session_folder_without_arm_csv(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="arm_left.csv"):
        PlaybackModule(str(tmp_path / "empty"))


def test_rejects_base_frame_only_session(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    for arm in ("arm_left", "arm_right"):
        with open(folder / f"{arm}.csv", "w") as f:
            f.write("time;wall_clock_ns;" + ";".join(f"O_T_EE_cmd_{i}" for i in range(16))
                    + ";gripper_cmd;state\n")
            f.write("0.0;0;" + ";".join("0.0" for _ in range(16)) + ";0.08;4\n")
    with pytest.raises(KeyError, match="world-frame"):
        PlaybackModule(str(folder))


def test_rejects_base_frame_only_episode(tmp_path):
    import h5py

    path = tmp_path / "old.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("observations/timestamp_ns", data=(np.arange(10) * 1e9 / RATE_HZ).astype(np.int64))
        for arm in ("arm_left", "arm_right"):
            f.create_dataset(f"observations/{arm}/state", data=np.full(10, 4))
            f.create_dataset(f"actions/{arm}/O_T_EE_cmd", data=np.zeros((10, 16)))
    with pytest.raises(KeyError, match="world-frame"):
        PlaybackModule(str(path))
