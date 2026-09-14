"""Online adapter around the trained ACT checkpoint (teleop-policy). Mirrors
teleop-policy/scripts/replay_utils.py's load_model/load_stats/load_image and
its ALOHA-style temporal-ensembling math exactly (all already validated by
eval_open_loop.py/verify_stage2.py) -- reimplemented here rather than
imported, since replay_utils.py itself uses bare `from dataset.transforms
import ...` / `from models.act.act import ACT` absolute imports that aren't
compatible with the teleop_policy-namespaced install (see pyproject.toml's
comment on why teleop-intent needed a metrics shim and teleop-policy didn't
-- this is the one place teleop-policy's flat-repo import convention leaks
into something we want to reuse). The only genuinely new logic is turning
open_loop_replay's whole-episode loop (builds the full `pending` dict, then
reads it out after the fact) into a step()-shaped rolling window: append
this tick's chunk prediction, ensemble immediately using whatever's
accumulated in this tick's slot so far, then evict it.

Does implement contracts.Module's shape (step(frame: SensorFrame) ->
ActionOutput), but does NOT read proprio from frame.proprio -- ACT's proprio
is pos(3) + 6D rotation (Zhou et al. 2019) + gripper_width, a different
representation from what intent models read out of SensorFrame.proprio (see
live_source.py's module docstring). PolicyModule is constructed with direct
references to the live ArmChannels instead, and only uses `frame` for camera
images (frame.images) and the tick's frame_id (chunk bookkeeping).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from teleop_orchestrator.contracts import ActionOutput, SensorFrame
from teleop_orchestrator.live.arm_channel import ArmChannel
from teleop_policy.dataset.transforms import denormalize, normalize, rot6d_to_matrix
from teleop_policy.models.act import act as _act_module
from teleop_policy.models.act.act import ACT

M_ENSEMBLE = 0.01  # ALOHA default: w_i ~ exp(-m*i), i = steps since prediction was made; matches replay_utils.py

# dataset_config's paths (e.g. normalize.stats_file: "dataset/stats.npz") are
# relative to teleop-policy's own repo root -- where train.py wrote the
# checkpoint from -- not wherever the orchestrator happens to be launched
# from. act.py's file location anchors us back to that root regardless of cwd.
_TELEOP_POLICY_ROOT = Path(_act_module.__file__).resolve().parents[2]


def _resolve(path_str: str) -> str:
    """Resolves a dataset_config path against teleop-policy's repo root if
    it isn't already absolute."""
    p = Path(path_str)
    return str(p if p.is_absolute() else _TELEOP_POLICY_ROOT / p)


def _load_model(checkpoint_path: str, device: torch.device):
    """Mirrors replay_utils.load_model exactly."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    dcfg, mcfg = ckpt["dataset_config"], ckpt["model_config"]
    n_cameras = len(dcfg["cameras"]["use"])
    action_dim = dcfg["action"]["dims_per_arm"] * len(dcfg["action"]["arms"])
    model = ACT(n_cameras=n_cameras, proprio_dim=action_dim, action_dim=action_dim,
                chunk_size=dcfg["action"]["chunk_size"], **mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dcfg


def _load_stats(dcfg: dict) -> dict:
    """Mirrors replay_utils.load_stats, resolving stats_file against
    teleop-policy's repo root (see _resolve) instead of cwd."""
    npz = np.load(_resolve(dcfg["normalize"]["stats_file"]))
    return {k: npz[k] for k in npz.files}


def _live_proprio(arm_state) -> np.ndarray:
    """pos(3) + rot6d(6, Zhou et al. -- first two columns of the rotation
    matrix, see dataset/transforms.py) + gripper_width(1) -- matches
    dataset.yaml's proprio composition (O_T_EE_world-derived). arm_state.
    position/quaternion (ArmStateMsg) is already world frame as published by
    the sim (T_base_ * O_T_EE) -- no transform needed here to match what the
    act_sorting_world checkpoint was trained on."""
    from scipy.spatial.transform import Rotation
    pos = np.asarray(arm_state.position, dtype=np.float32)
    w, x, y, z = arm_state.quaternion
    r = Rotation.from_quat([x, y, z, w]).as_matrix()
    rot6d = np.concatenate([r[:, 0], r[:, 1]])
    return np.concatenate([pos, rot6d, [arm_state.gripper_width]]).astype(np.float32)


def _rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """Inverse of the rotation part of _live_proprio, for ActionOutput.ee_pose."""
    from scipy.spatial.transform import Rotation
    r = rot6d_to_matrix(rot6d)
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return np.array([w, x, y, z])


def _preprocess_image(img: np.ndarray, hw: tuple, stored_hw: Optional[tuple] = None) -> np.ndarray:
    """Resizes to (h, w) and scales to [0, 1] -- matches replay_utils.load_image.

    stored_hw is the resolution the TRAINING images were archived at, which is
    not necessarily dcfg["cameras"]["resize_to"]. episode.hdf5 was written with
    manifest image_scale=0.25, so:

        head_cam_left    1280x960 source -> stored 320x240 -> resize_to 320x240  (no-op)
        wrist_cam_*       640x480 source -> stored 160x120 -> resize_to 320x240  (2x UPSCALE)

    and cv2.INTER_AREA degenerates to nearest-neighbour when upscaling, so the
    wrist backbones were trained on images in which every 2x2 block is exactly
    constant -- 160x120 of real information in a 320x240 tensor (measured: 2x2
    block-constant fraction 1.000 on wrist, 0.069 on head).

    Live, the wrist shm streams are 640x480 (teleop-simulator
    pipeline_config_local.yaml stream_cameras), so a single resize to 320x240 is
    a 2x DOWNscale and hands those two backbones genuine 320x240 detail --
    measured block-constant 0.31/0.20 instead of 1.00. Same tensor shape, ~4x
    the spatial-frequency content, on 2 of 3 cameras.

    So when stored_hw is smaller than hw we reproduce the training degradation
    exactly: source -> stored_hw -> hw, both hops INTER_AREA, same as the
    offline path. Passing stored_hw=None keeps the old single-resize behaviour.
    """
    if stored_hw is not None and tuple(stored_hw) != tuple(hw):
        if img.shape[:2] != tuple(stored_hw):
            img = cv2.resize(img, (stored_hw[1], stored_hw[0]), interpolation=cv2.INTER_AREA)
    if img.shape[:2] != tuple(hw):
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


class PolicyModule:
    """Runs ACT online: one K-step chunk prediction per tick, ALOHA-style
    temporally-ensembled against overlapping predictions from recent ticks.
    reset()/step() shape matches contracts.Module; see module docstring for
    why it isn't formally SensorModule[ActionOutput] (proprio source)."""

    def __init__(self, checkpoint_path: str, arm_left: ArmChannel, arm_right: ArmChannel,
                 device: Optional[str] = None, debug_dump_dir: Optional[str] = None,
                 debug_dump_every: int = 30,
                 stored_hw: Optional[dict] = None,
                 timing_every: int = 0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Say this out loud. A CPU-only torch build silently falls back here,
        # and the only symptom is that the whole loop runs several times slower
        # than it should -- which is easy to misread as "the sim is heavy".
        # 3x ResNet-18 at 240x320 is single-digit ms on a discrete GPU and
        # ~100 ms on CPU, so this one line decides the achievable tick rate.
        if self.device.type == "cuda":
            print(f"[PolicyModule] device=cuda ({torch.cuda.get_device_name(0)})")
        else:
            print(f"[PolicyModule] device={self.device.type} -- torch.cuda.is_available() is False. "
                  f"Inference will be ~10x slower than on GPU and will saturate several CPU cores. "
                  f"If this machine has a usable GPU, the installed torch is a CPU-only build.")
        self.model, self.dcfg = _load_model(checkpoint_path, self.device)
        self.stats = _load_stats(self.dcfg)
        self._arm_left = arm_left
        self._arm_right = arm_right
        # Camera identity AND order come from the checkpoint itself, not a
        # caller-supplied list: model.backbones[i] is one ResNet per camera,
        # order-sensitive, and must exactly match what it was trained with.
        # frame.images must have these exact keys -- see system.yaml's
        # network.cameras (keys must match dcfg["cameras"]["use"] names).
        self._camera_names = self.dcfg["cameras"]["use"]
        self._hw = tuple(self.dcfg["cameras"]["resize_to"])
        # Per-camera archived training resolution -- see _preprocess_image.
        # Not derivable from the checkpoint (dcfg records resize_to, not the
        # resolution episode.hdf5 was written at), so it comes from config;
        # unset means "same as resize_to", i.e. the old single-resize path.
        self._stored_hw = {name: tuple(hw) for name, hw in (stored_hw or {}).items()}
        missing = [c for c in self._camera_names if c not in self._stored_hw]
        if missing:
            print(f"[PolicyModule] WARNING: no stored_hw for {missing} -- assuming the "
                  f"training images were archived at resize_to={self._hw}. If they were "
                  f"archived smaller (manifest image_scale), these backbones will see "
                  f"sharper input than they were trained on.")
        self._K = self.dcfg["action"]["chunk_size"]
        self._d = self.dcfg["action"]["dims_per_arm"]
        self._pending: dict = {}
        # Training action-grid rate; the unit chunk offsets are expressed in.
        # See _slot_index for why this and not the source frame rate.
        self._rate_hz = float(self.dcfg["alignment"]["rate_hz"])
        self._t0_ns: Optional[int] = None
        # Debug-only: periodically writes exactly what each camera name
        # resolves to (post-resize, i.e. the literal network input) as a PNG
        # -- for confirming camera identity/order hasn't silently drifted
        # (we've hit that bug once already, see run.py's build_modules
        # comment). None (default) disables this entirely, zero overhead.
        self._debug_dump_dir = debug_dump_dir
        self._debug_dump_every = debug_dump_every
        self._tick = 0
        if self._debug_dump_dir is not None:
            os.makedirs(self._debug_dump_dir, exist_ok=True)
        # Rolling per-stage latency accumulators; see _record_timing.
        self._timing_every = timing_every
        self._t_pre = self._t_fwd = self._t_ens = 0.0
        self._t_n = 0

    def reset(self) -> None:
        """Clears the rolling chunk-prediction buffer at an episode boundary."""
        self._pending.clear()
        self._t0_ns = None  # re-anchor the training-grid slot index (see _slot_index)

    def _record_timing(self, t0: float, t_pre: float, t_fwd: float, t_end: float) -> None:
        """Accumulates per-stage step() latency and prints a breakdown every
        timing_every ticks.

        Off by default (timing_every=0); pass timing_every=100 to switch it
        back on. Kept rather than deleted because "the loop is slow and the CPU
        is busy" has at least three unrelated explanations -- image
        preprocessing on the CPU, the forward pass falling back to CPU, and the
        source simply not publishing faster -- and they need completely
        different fixes. Attributing the time is cheaper than guessing, and it
        is how the CPU-only torch build was found. Note the forward figure is
        only meaningful on CUDA if you account for async launch: the .cpu()
        call on the next line forces a sync, so t_fwd does include real GPU
        time.
        """
        self._t_pre += t_pre - t0
        self._t_fwd += t_fwd - t_pre
        self._t_ens += t_end - t_fwd
        self._t_n += 1
        if self._timing_every and self._t_n >= self._timing_every:
            n = self._t_n
            total = (self._t_pre + self._t_fwd + self._t_ens) / n
            print(f"[PolicyModule] {n} ticks | preprocess {self._t_pre / n * 1e3:6.1f} ms | "
                  f"forward {self._t_fwd / n * 1e3:6.1f} ms | ensemble {self._t_ens / n * 1e3:5.1f} ms | "
                  f"total {total * 1e3:6.1f} ms ({1.0 / max(total, 1e-9):5.1f} Hz ceiling) | "
                  f"pending slots {len(self._pending)}")
            self._t_pre = self._t_fwd = self._t_ens = 0.0
            self._t_n = 0

    def step(self, frame: SensorFrame) -> ActionOutput:
        left, right = self._arm_left.latest(), self._arm_right.latest()
        if left is None or right is None or any(c not in frame.images for c in self._camera_names):
            return ActionOutput(ee_pose={}, gripper={}, extras={"skipped": "arm state or camera unavailable"})

        t_start = time.perf_counter()
        images = np.stack([_preprocess_image(frame.images[c], self._hw, self._stored_hw.get(c))
                           for c in self._camera_names])
        self._maybe_dump_debug_images(images)
        images_t = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)

        proprio = np.concatenate([_live_proprio(left), _live_proprio(right)])
        proprio_n = normalize(proprio, self.stats["proprio_mean"], self.stats["proprio_std"])
        proprio_t = torch.from_numpy(proprio_n).unsqueeze(0).float().to(self.device)
        t_pre = time.perf_counter()

        with torch.no_grad():
            a_hat, _, _ = self.model(images_t, proprio_t)  # z=0 at inference (actions=None)
        a_hat = denormalize(a_hat[0].cpu().numpy(), self.stats["action_mean"], self.stats["action_std"])
        t_fwd = time.perf_counter()

        action = self._ensemble(self._slot_index(frame.timestamp_ns), a_hat)
        self._record_timing(t_start, t_pre, t_fwd, time.perf_counter())

        d = self._d
        left_action, right_action = action[:d], action[d:2 * d]
        return ActionOutput(
            ee_pose={
                "arm_left": np.concatenate([left_action[:3], _rot6d_to_quat_wxyz(left_action[3:9])]),
                "arm_right": np.concatenate([right_action[:3], _rot6d_to_quat_wxyz(right_action[3:9])]),
            },
            gripper={"arm_left": float(left_action[9]), "arm_right": float(right_action[9])},
            # Raw chunk endpoints, before ensembling. Everything logged so far
            # has been the ensembled command, which cannot distinguish "the
            # model is asking to descend and execution isn't getting there"
            # from "the model is asking to hover". a_hat[0] is what it wants
            # now; a_hat[K-1] is where it intends to be ~2s ahead. If the
            # policy hovers 25mm high while chunk_z_last sits 25mm lower, the
            # intent is right and the loop is losing it; if chunk_z_last is
            # flat at the hover height, the model genuinely wants to hover and
            # the problem is upstream in what it learned.
            extras={
                "chunk_z_first_left": float(a_hat[0][2]),
                "chunk_z_last_left": float(a_hat[-1][2]),
                "chunk_z_first_right": float(a_hat[0][d + 2]),
                "chunk_z_last_right": float(a_hat[-1][d + 2]),
                "chunk_grip_last_left": float(a_hat[-1][9]),
                "chunk_grip_last_right": float(a_hat[-1][d + 9]),
            },
        )

    def _maybe_dump_debug_images(self, images: np.ndarray) -> None:
        """Writes one PNG per camera, every debug_dump_every ticks, showing
        exactly the array each backbone actually receives (post-resize,
        pre-normalize) -- eyeball these against what you expect each camera
        name to show. No-op unless debug_dump_dir was set."""
        if self._debug_dump_dir is None:
            return
        if self._tick % self._debug_dump_every == 0:
            for name, img in zip(self._camera_names, images):
                bgr = (img[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
                out_path = os.path.join(self._debug_dump_dir, f"{self._tick:06d}_{name}.png")
                cv2.imwrite(out_path, bgr)
        self._tick += 1

    def _slot_index(self, timestamp_ns: int) -> int:
        """Maps a tick's wall-clock timestamp onto the TRAINING action grid.

        a_hat[i] is "the action i steps ahead at dataset.yaml's
        alignment.rate_hz" -- i/30 s, not i source-frames. Keying the pending
        buffer by frame.frame_id (the old behaviour) only coincides with that
        if the source publishes at exactly the training rate. It publishes at
        ~20 Hz, so every chunk was being played back over 3.0 s instead of
        2.0 s: a uniform 1.5x slow-motion on every commanded trajectory.

        Anchoring on the training grid instead makes the mapping rate-agnostic
        -- the source can publish at any rate, and each prediction still lands
        in the slot matching the wall-clock instant it was predicted for. At
        20 Hz the slot index advances by 1.5 per tick, so it strides 1,2,1,2...
        and the skipped slots are simply never read (the eviction below reaps
        them).

        Anchored at the first tick rather than the raw epoch so the index stays
        small enough for float rounding to be exact.
        """
        if self._t0_ns is None:
            self._t0_ns = timestamp_ns
        return int(round((timestamp_ns - self._t0_ns) * 1e-9 * self._rate_hz))

    def _ensemble(self, t: int, a_hat: np.ndarray) -> np.ndarray:
        """Appends this tick's K-step chunk prediction into the rolling
        buffer, then ensembles and evicts slot t (matches
        replay_utils.open_loop_replay's math, spread over real-time ticks
        instead of a single batched loop).

        t is a training-grid slot index (see _slot_index), NOT a frame_id, so
        chunk offset i and slot offset i mean the same duration.
        """
        for i in range(self._K):
            self._pending.setdefault(t + i, []).append(a_hat[i])
        preds = np.stack(self._pending.pop(t, [a_hat[0]]))

        # frame_id does not advance by exactly 1 per tick (measured stride 2-3,
        # since the policy ticks slower than the source publishes), so every
        # slot the stride skipped over is never popped by the line above. Those
        # entries used to live in the dict for the rest of the session: ~K
        # inserts vs ~K/stride pops per tick, i.e. a few hundred orphaned
        # 20-float arrays per second, unbounded. Nothing behind us can ever be
        # consumed again, so drop it.
        if len(self._pending) > 2 * self._K:
            for key in [k for k in self._pending if k < t]:
                del self._pending[key]

        ages = np.arange(len(preds))[::-1]  # 0 = most recent prediction
        w = np.exp(-M_ENSEMBLE * ages)
        w /= w.sum()
        return (w[:, None] * preds).sum(axis=0)
