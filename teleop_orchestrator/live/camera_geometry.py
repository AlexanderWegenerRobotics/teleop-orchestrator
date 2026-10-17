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
import os
import time
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
                          intrinsics: CameraIntrinsics, scale_u: float = 0.5, scale_v: float = 1.0) -> tuple:
    """Raw GazeSampleMsg pixels -> normalized ray coords, matching IntentionBuffer::fuseGaze.

    The interface reports gaze in pixels of the video texture it displays. scale_u/scale_v map those
    onto the intrinsics' single-camera image: 0.5 for the stereo head stream (2560 wide, one eye is
    1280), 1.0 for a mono stream such as the desk camera. Same as the avatar's gaze_scale_u/v.
    """
    gaze_u_cam = gaze_px_x_wire * scale_u
    gaze_v_cam = gaze_px_y_wire * scale_v
    return (gaze_u_cam - intrinsics.cx) / intrinsics.fx, (gaze_v_cam - intrinsics.cy) / intrinsics.fy


_LAYOUT_RATIOS = (1.0, 0.5)
_LAYOUT_TOL = 0.02


class CameraParams:
    """One camera's camera_params.json entry, re-read whenever the sim rewrites the file, plus gaze normalization."""

    def __init__(self, path: str, camera_name: str, stereo_stream: bool, recheck_s: float = 1.0):
        self.path = path
        self.camera_name = camera_name
        self.stereo_stream = stereo_stream
        self._recheck_s = recheck_s
        self._next_check = 0.0
        self._mtime = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.T_world_cam: Optional[np.ndarray] = None
        self._warned: set = set()
        if not self._reload(force=True):
            raise KeyError(f"{camera_name!r} not readable from {path}")

    def _reload(self, force: bool = False) -> bool:
        """Re-reads the entry if the file's mtime changed; keeps the previous values on a partial or bad read."""
        try:
            mtime = os.stat(self.path).st_mtime_ns
            if mtime == self._mtime and not force:
                return True
            with open(self.path) as f:
                c = json.load(f)[self.camera_name]
            new = CameraIntrinsics(fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"], width=c["width"], height=c["height"])
        except (OSError, ValueError, KeyError) as e:
            if force:
                raise
            print(f"[camera] could not re-read {self.path} ({e}); keeping the previous {self.camera_name} parameters")
            return False
        old = self.intrinsics
        self.intrinsics = new
        self.T_world_cam = np.asarray(c["T_world_cam"], dtype=np.float64) if "T_world_cam" in c else None
        self._mtime = mtime
        was = f" (was {old.width:g}x{old.height:g})" if old is not None else ""
        print(f"[camera] {self.camera_name}: {new.width:g}x{new.height:g}, fx {new.fx:.1f}{was} from {self.path}")
        return True

    def refresh(self) -> None:
        """Cheap per-tick check; stats the file at most once per recheck_s."""
        now = time.monotonic()
        if now >= self._next_check:
            self._next_check = now + self._recheck_s
            self._reload()

    def _warn_once(self, key, msg: str) -> None:
        """Prints msg the first time key is seen."""
        if key not in self._warned:
            self._warned.add(key)
            print(msg)

    def gaze_scale(self, image_w: int = 0, image_h: int = 0) -> Optional[tuple]:
        """Factor from interface texture pixels to camera pixels, or None when the two sizes cannot describe the same image."""
        cam = self.intrinsics
        if image_w <= 0 or image_h <= 0:
            layout = "side-by-side stereo" if self.stereo_stream else "mono"
            self._warn_once("no_size", f"[gaze] interface does not report its texture size (old build); "
                                       f"assuming a {layout} stream of {cam.width:g}x{cam.height:g}")
            return (0.5, 1.0) if self.stereo_stream else (1.0, 1.0)
        su, sv = cam.width / image_w, cam.height / image_h
        if not any(abs(su / sv - r) <= _LAYOUT_TOL * r for r in _LAYOUT_RATIOS):
            self._warn_once((image_w, image_h, cam.width, cam.height),
                            f"[gaze] WARNING: interface texture {image_w}x{image_h} is not a mono or side-by-side "
                            f"stereo view of {self.camera_name} {cam.width:g}x{cam.height:g} ({self.path}). "
                            "Gaze is DROPPED until they agree: a stream crop, or camera_params.json from another sim config.")
            return None
        return su, sv

    def gaze_ray(self, px_x: float, px_y: float, image_w: int = 0, image_h: int = 0) -> Optional[tuple]:
        """Interface gaze pixels -> normalized ray coords, or None if the texture and camera sizes disagree."""
        scale = self.gaze_scale(image_w, image_h)
        if scale is None:
            return None
        return normalize_gaze_pixel(px_x, px_y, self.intrinsics, *scale)
