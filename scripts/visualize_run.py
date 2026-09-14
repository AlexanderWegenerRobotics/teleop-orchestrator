"""Overlays a saved run.hdf5's predicted world-frame EE targets onto the
head camera's recorded video, so you can see what the policy was aiming for
frame-by-frame -- useful on its own for sanity-checking (no video needed for
that, see plot_run.py) but especially for actually watching what happened.

Only the head camera is supported: camera_geometry.project_to_ray needs a
head_pan/head_tilt + fixed head->camera offset, which is exactly the chain
already built for gaze/candidate projection (see camera_geometry.py). Wrist
cameras move with the arm itself and would need new forward-kinematics-based
extrinsics (arm base -> wrist camera mount) that don't exist anywhere in this
codebase yet -- flagging that rather than guessing at it.

Assumes a fixed head_pan/head_tilt for the whole session (true for an
autonomous run using run.py's head_look_down_offset heuristic, since that's
constant every tick -- NOT true for a VR-teleop session with a moving head,
don't point this at one of those without passing --head-tilt per-segment
yourself). Reuses teleop-intent's video decoder (ffmpeg-based) rather than
reimplementing it -- sys.path insert, not an installed dependency (scripts/
isn't part of that repo's installable package, see its pyproject.toml).

Usage:
    python scripts/visualize_run.py logs/run_20260806_193000.hdf5 \\
        /path/to/head_cam_stereo.h264 out.mp4 \\
        --head-tilt -0.6 --stereo-layout vertical
    (if the overlay lands on the wrong eye's video, add --eye second)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import warnings

import cv2
import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "teleop-intent", "scripts"))
import episode_to_hdf5 as _e2h  # noqa: E402 -- sys.path set above, see module docstring
from episode_to_hdf5 import load_video  # noqa: E402


@contextlib.contextmanager
def _tolerant_sidecar_parsing():
    """Lets load_video survive a truncated .timestamps.csv.

    The sidecar is written incrementally during recording, so killing the sim
    (or a crash -- see avatar_crash.log) leaves the final row half-written:
    observed '2531,1786228567140069300,17862285671047' with no newline. The
    upstream loader parses it with a bare np.genfromtxt, which aborts the whole
    file on one ragged row -- so a 2532-frame recording becomes unreadable
    because of its last frame.

    numpy already has the right behaviour behind a flag, so rather than
    duplicating load_video here (it also owns ffprobe/decode, see this module's
    docstring on not reimplementing the decoder) we just default that flag on
    for the duration of the call. Dropping the partial row is safe: the sidecar
    only supplies wall-clock timestamps, and load_video already falls back to
    the markers encoded in the video rows for any frame the sidecar doesn't
    cover.
    """
    real = _e2h.np.genfromtxt

    def tolerant(*args, **kwargs):
        kwargs.setdefault("invalid_raise", False)
        with warnings.catch_warnings():
            # numpy warns once per skipped line; the count is reported below.
            warnings.simplefilter("ignore")
            return real(*args, **kwargs)

    _e2h.np.genfromtxt = tolerant
    try:
        yield
    finally:
        _e2h.np.genfromtxt = real


def _sidecar_health(video_path):
    """(n_good, n_ragged) for the sidecar, or None if there isn't one."""
    sidecar = os.path.splitext(video_path)[0] + ".timestamps.csv"
    if not os.path.exists(sidecar):
        return None
    good = ragged = 0
    with open(sidecar) as f:
        header = f.readline().rstrip("\n").split(",")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if len(line.split(",")) == len(header):
                good += 1
            else:
                ragged += 1
    return good, ragged

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from teleop_orchestrator.live.camera_geometry import load_intrinsics, project_to_ray  # noqa: E402

ARM_COLORS = {"arm_left": (60, 60, 255), "arm_right": (255, 140, 60)}  # BGR, per arm
TRAIL_LEN = 15  # ticks of trailing marker history drawn per arm, for visible motion


def _draw_trail(img, trail, color, scale):
    """Draws the trail as a smooth polyline that fades and thins toward the
    older end -- each segment alpha-blended individually (cropped to its own
    bounding box, not the whole frame, so this stays cheap) rather than drawn
    as flat opaque dots, so it reads as motion leading into the current
    marker instead of a string of beads."""
    if len(trail) < 2:
        return
    n = len(trail) - 1
    for j in range(n):
        p1, p2 = trail[j], trail[j + 1]
        age_frac = (j + 1) / n  # 0 = oldest segment, 1 = newest (right before the current marker)
        alpha = 0.15 + 0.55 * age_frac
        thickness = max(1, int(round((1 + 4 * age_frac) * scale)))

        pad = thickness * 3 + 2
        x0, x1 = sorted((p1[0], p2[0]))
        y0, y1 = sorted((p1[1], p2[1]))
        x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        x1, y1 = min(x1 + pad, img.shape[1]), min(y1 + pad, img.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue

        roi = img[y0:y1, x0:x1]
        overlay = roi.copy()
        cv2.line(overlay, (p1[0] - x0, p1[1] - y0), (p2[0] - x0, p2[1] - y0),
                  color, thickness, cv2.LINE_AA)
        img[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


def _draw_arm_marker(img, arm, trail, color, scale):
    """Draws one arm's current target as a crosshair (white outline + colored
    fill so it reads over any background color, not just a small dot) plus a
    labeled background box, and the recent trail as a fading polyline behind
    it (see _draw_trail). scale accounts for output resolutions other than
    the ~960px this was tuned at (e.g. a small synthetic test video)."""
    if not trail:
        return
    u, v = trail[-1]
    cross = max(1, int(8 * scale))
    thick_outline = max(1, int(3 * scale))
    thick_fill = max(1, int(1 * scale))
    ring_r = max(1, int(6 * scale))

    _draw_trail(img, trail, color, scale)

    # white outline first (contrast against any background), colored crosshair + ring on top
    cv2.line(img, (u - cross, v), (u + cross, v), (255, 255, 255), thick_outline, cv2.LINE_AA)
    cv2.line(img, (u, v - cross), (u, v + cross), (255, 255, 255), thick_outline, cv2.LINE_AA)
    cv2.circle(img, (u, v), ring_r, (255, 255, 255), thick_outline, cv2.LINE_AA)
    cv2.line(img, (u - cross, v), (u + cross, v), color, thick_fill, cv2.LINE_AA)
    cv2.line(img, (u, v - cross), (u, v + cross), color, thick_fill, cv2.LINE_AA)
    cv2.circle(img, (u, v), ring_r, color, thick_fill, cv2.LINE_AA)

    font_scale = 0.7 * scale
    font_thick = max(1, int(2 * scale))
    (tw, th), _ = cv2.getTextSize(arm, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    label_org = (u + cross + 4, v - cross)
    pad = max(2, int(3 * scale))
    cv2.rectangle(img, (label_org[0] - pad, label_org[1] - th - pad),
                  (label_org[0] + tw + pad, label_org[1] + pad), (0, 0, 0), -1)
    cv2.putText(img, arm, label_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)


def nearest_idx(stream_ts, grid_ts):
    """For each grid time, index of the nearest stream sample -- same as
    episode_to_hdf5.py's version."""
    pos = np.searchsorted(stream_ts, grid_ts)
    pos = np.clip(pos, 1, len(stream_ts) - 1)
    left, right = stream_ts[pos - 1], stream_ts[pos]
    return np.where(np.abs(grid_ts - left) <= np.abs(right - grid_ts), pos - 1, pos)


def load_run(path):
    with h5py.File(path, "r") as f:
        t_ns = f["frames/timestamp_ns"][:]
        modules = [m.decode() if isinstance(m, bytes) else m for m in f.attrs["modules"]]
        policy_name = next((m for m in modules if f"modules/{m}/arm_left/ee_pose" in f), None)
        if policy_name is None:
            raise SystemExit(f"no policy module with ee_pose found in {path} (modules: {modules})")
        positions = {}
        for arm in ("arm_left", "arm_right"):
            key = f"modules/{policy_name}/{arm}/ee_pose"
            if key in f:
                positions[arm] = f[key][:, :3]
        return t_ns, positions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_hdf5")
    ap.add_argument("video_path", help="head camera .h264 elementary stream")
    ap.add_argument("out_path", help="annotated .mp4 to write")
    ap.add_argument("--system-config", default="configs/system.yaml")
    ap.add_argument("--robot-config", default="../teleop-simulator/config/robot_config_local.yaml",
                     help="head.q0 lives here, not system.yaml -- head_control.cpp adds the command "
                          "(head_look_down_offset) to q0_ to get the actual joint angle, and "
                          "project_to_ray needs that final absolute angle, not the offset alone")
    ap.add_argument("--head-pan", type=float, default=None,
                     help="overrides the final absolute head pan (q0.pan + head_look_down_offset.pan), not just the offset")
    ap.add_argument("--head-tilt", type=float, default=None,
                     help="overrides the final absolute head tilt (q0.tilt + head_look_down_offset.tilt), not just the offset")
    ap.add_argument("--stereo-layout", choices=["none", "horizontal", "vertical"], default="none",
                     help="none = video is already single-eye; horizontal = side-by-side "
                          "(left eye in x[0,W/2)); vertical = stacked (top/bottom halves)")
    ap.add_argument("--eye", choices=["first", "second"], default="first",
                     help="which half to keep for stereo-layout horizontal/vertical -- 'first' is "
                          "left/top depending on layout; if the overlay ends up on the wrong eye's "
                          "video, just pass --eye second instead of guessing the rig's convention")
    ap.add_argument("--scale", type=float, default=1.0, help="decode scale, 1.0 = native resolution")
    args = ap.parse_args()

    with open(args.system_config) as f:
        cfg = yaml.safe_load(f)
    geo = cfg["geometry"]
    head_position = tuple(geo["head_position"])
    camera_position = tuple(geo["camera_position"])
    intrinsics = load_intrinsics(geo["camera_params_path"], geo["camera_name"])

    if args.head_pan is not None and args.head_tilt is not None:
        head_pan, head_tilt = args.head_pan, args.head_tilt
    else:
        with open(args.robot_config) as f:
            robot_cfg = yaml.safe_load(f)
        head_device = next(d for d in robot_cfg["devices"] if d["name"] == "head")
        q0_pan, q0_tilt = head_device["q0"]
        offset = geo["head_look_down_offset"]
        head_pan = args.head_pan if args.head_pan is not None else q0_pan + offset["pan"]
        head_tilt = args.head_tilt if args.head_tilt is not None else q0_tilt + offset["tilt"]
    print(f"[visualize_run] using fixed absolute head_pan={head_pan} head_tilt={head_tilt} "
          f"(head.q0 from {args.robot_config} + geometry.head_look_down_offset from {args.system_config} "
          f"-- pass --head-pan/--head-tilt to override the absolute value directly)")

    t_ns, positions = load_run(args.run_hdf5)
    if not positions:
        raise SystemExit("run.hdf5 has no arm ee_pose data to visualize")

    if not os.path.exists(args.video_path):
        raise SystemExit(f"no such file: {args.video_path}")
    if os.path.getsize(args.video_path) == 0:
        raise SystemExit(f"{args.video_path} is empty (0 bytes) -- nothing was recorded to it")
    health = _sidecar_health(args.video_path)
    if health and health[1]:
        print(f"[visualize_run] sidecar has {health[1]} malformed row(s) alongside {health[0]} good "
              f"ones -- almost certainly a recording interrupted mid-write. Skipping them; "
              f"those frames fall back to the timestamps encoded in the video itself.")
    try:
        with _tolerant_sidecar_parsing():
            frames, wall_ns, _fids = load_video(args.video_path, args.scale)
    except subprocess.CalledProcessError as e:
        # probe_dims/decode_frames use check=True and swallow ffprobe's own
        # error text -- surface it, since "nonzero exit" alone isn't
        # actionable (wrong container, corrupt file, codec ffprobe can't
        # parse as raw h264, etc. all look identical without this).
        raise SystemExit(f"ffprobe failed on {args.video_path}:\n{e.stderr or '(no stderr captured)'}")

    if args.stereo_layout != "none":
        h, w = frames.shape[1:3]
        if args.stereo_layout == "horizontal":
            half = w // 2
            frames = frames[:, :, :half, :] if args.eye == "first" else frames[:, :, half:, :]
        else:  # vertical
            half = h // 2
            frames = frames[:, :half, :, :] if args.eye == "first" else frames[:, half:, :, :]

    sel = nearest_idx(t_ns, wall_ns)  # per video frame, index into run.hdf5 ticks

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames.shape[1:3]
    marker_scale = max(w, h) / 960.0  # pixel sizes below are tuned for ~960px frames; scale for other resolutions
    print(f"[visualize_run] output frame size: {w}x{h}  intrinsics: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
          f"cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f} (calibrated for {intrinsics.width:.0f}x{intrinsics.height:.0f}) "
          f"-- if that calibrated size doesn't match the output frame size above, the intrinsics don't "
          f"apply to this crop and every projection below will be wrong.")
    writer = cv2.VideoWriter(args.out_path, fourcc, 30.0, (w, h))

    trails = {arm: [] for arm in positions}
    n_in_front = {arm: 0 for arm in positions}
    n_in_bounds = {arm: 0 for arm in positions}
    uv_range = {arm: [np.inf, -np.inf, np.inf, -np.inf] for arm in positions}  # umin,umax,vmin,vmax
    for i in range(len(frames)):
        img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR).copy()
        t = sel[i]
        for arm, pos_arr in positions.items():
            ray = project_to_ray(pos_arr[t], head_pan, head_tilt, head_position, camera_position)
            color = ARM_COLORS[arm]
            if ray is not None:
                u = int(round(intrinsics.fx * ray[0] + intrinsics.cx))
                v = int(round(intrinsics.fy * ray[1] + intrinsics.cy))
                n_in_front[arm] += 1
                r = uv_range[arm]
                uv_range[arm] = [min(r[0], u), max(r[1], u), min(r[2], v), max(r[3], v)]
                if 0 <= u < w and 0 <= v < h:
                    n_in_bounds[arm] += 1
                    trails[arm].append((u, v))
                    trails[arm] = trails[arm][-TRAIL_LEN:]
                # else: projected behind the visible frame -- deliberately NOT
                # clamped/drawn at the edge, since a clamped marker looks
                # identical to a correct one and would hide exactly the bug
                # this diagnostic is trying to catch.
            _draw_arm_marker(img, arm, trails[arm], color, marker_scale)
        writer.write(img)

    writer.release()
    for arm in positions:
        n = max(len(frames), 1)
        umin, umax, vmin, vmax = uv_range[arm]
        print(f"[visualize_run] {arm}: in front of camera {n_in_front[arm] / n:.1%} of frames, "
              f"but only {n_in_bounds[arm] / n:.1%} landed inside the {w}x{h} visible frame "
              f"(projected pixel range: u=[{umin:.0f},{umax:.0f}] v=[{vmin:.0f},{vmax:.0f}])")
    print(f"[ok] {args.out_path}")


if __name__ == "__main__":
    main()
