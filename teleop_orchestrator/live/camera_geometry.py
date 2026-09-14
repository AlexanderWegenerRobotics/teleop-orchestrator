"""Head-mounted camera projection, ported from
teleop-simulator/src/intention/intention_buffer.cpp (IntentionBuffer::
projectToImage + the gaze normalization in fuseGaze) now that gaze fusion
runs in the orchestrator instead of avatar.cpp's IntentionBuffer. Two
separate things live here, with different intrinsics needs:

  - Candidate/EE world positions -> normalized ray coords (px_u, px_v):
    pure geometry, no camera intrinsics needed -- see project_to_ray's
    docstring for why the algebra cancels them out.
  - Raw operator gaze pixels -> normalized ray coords: DOES need real
    fx/fy/cx/cy, since a measured pixel can only be turned into a ray
    with the camera's actual intrinsics. Read once from camera_params.json
    (written by Avatar::writeCameraParams at sim startup).

Both outputs land in the same "fraction of native camera frame" ray
convention CANDIDATE_FEATURE_NAMES/GLOBAL_FEATURE_NAMES already assume
(see contracts/features.py) -- this module exists so live and offline
(training-time) features can't drift on that convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Remaps from robot body frame to OpenCV frame: body X -> CV Z (forward),
# body Y -> CV -X (left), body Z -> CV -Y (down). Fixed, matches
# intention_buffer.cpp's R_body2cv exactly.
_R_BODY2CV = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: float
    height: float


def load_intrinsics(camera_params_path: str, camera_name: str) -> CameraIntrinsics:
    """Reads one camera's intrinsics out of camera_params.json (see
    Avatar::writeCameraParams). Only used for gaze normalization -- candidate
    projection doesn't need it, see project_to_ray."""
    with open(camera_params_path) as f:
        params = json.load(f)
    if camera_name not in params:
        raise KeyError(f"{camera_name!r} not in {camera_params_path} "
                        f"(available: {list(params)}) -- camera_params.json is written once at sim "
                        "startup for whatever cameras are in stream_cameras, re-run the sim if stale")
    c = params[camera_name]
    return CameraIntrinsics(fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"], width=c["width"], height=c["height"])


def _r_ch(head_pan: float, head_tilt: float) -> np.ndarray:
    """World-to-head rotation: tilt around +Y, pan world->head requires -pan
    (passive rotation) -- matches intention_buffer.cpp's fuseGaze exactly."""
    cp, sp = np.cos(-head_pan), np.sin(-head_pan)
    r_pan = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
    ct, st = np.cos(head_tilt), np.sin(head_tilt)
    r_tilt = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    return r_tilt @ r_pan


def project_to_ray(p_world, head_pan: float, head_tilt: float,
                    head_position, camera_position) -> Optional[tuple]:
    """World position -> normalized ray coords (px_u, px_v), or None if
    behind the camera. Matches projectToImage's chain (p_H = R_CH*(p_world -
    head_position); p_C = p_H - camera_position; p_CV = R_body2cv @ p_C) up
    to computing raw pixels (u=fx*x/z+cx) and immediately normalizing
    ((u-cx)/fx = x/z) -- fx/cx cancel algebraically, so this returns
    p_CV.x/p_CV.z, p_CV.y/p_CV.z directly. No intrinsics parameter is not a
    bug: a 3D ray's slope doesn't depend on the camera that happens to be
    looking at it, only on where the camera actually is.
    """
    p_h = _r_ch(head_pan, head_tilt) @ (np.asarray(p_world, dtype=np.float64) - np.asarray(head_position, dtype=np.float64))
    p_c = p_h - np.asarray(camera_position, dtype=np.float64)
    p_cv = _R_BODY2CV @ p_c
    if p_cv[2] <= 0.0:
        return None
    return float(p_cv[0] / p_cv[2]), float(p_cv[1] / p_cv[2])


def normalize_gaze_pixel(gaze_px_x_wire: float, gaze_px_y_wire: float,
                          intrinsics: CameraIntrinsics) -> tuple:
    """Raw GazeSampleMsg pixels -> normalized ray coords, matching
    IntentionBuffer::fuseGaze exactly: gaze_px_x arrives as GazeUV.X * 2560
    (full stereo width) from the UE5/HTC Vive Pro Eye bridge -- unchanged by
    routing gaze to the orchestrator instead of avatar, only the destination
    port changed, so this halving is still required to align into
    projectToImage's single-camera (cx=640, width=1280) coordinate space.
    """
    gaze_u_cam = gaze_px_x_wire * 0.5
    gaze_v_cam = gaze_px_y_wire
    return (gaze_u_cam - intrinsics.cx) / intrinsics.fx, (gaze_v_cam - intrinsics.cy) / intrinsics.fy
