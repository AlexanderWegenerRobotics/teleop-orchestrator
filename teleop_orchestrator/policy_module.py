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
from teleop_orchestrator.live.head_channel import HeadChannel
from teleop_orchestrator.live.object_source import AUTHORITY_POLICY
# Imported rather than recomputed. These two functions are the single
# definition of how wide the proprio/action vectors are and whether the head
# is part of them; duplicating that arithmetic here is exactly how this file
# ended up 20-dim while the checkpoint had moved to 22.
from teleop_policy.dataset.teleop_dataset import head_cfg, vector_dim
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
    # vector_dim() reads dcfg's own head block, so a head-enabled checkpoint
    # builds a 22-wide model and a head-free one still builds 20. The dcfg is
    # the one baked into the checkpoint, not whatever configs/ says today,
    # which is what makes an old checkpoint keep loading after the config moved
    # on. load_state_dict is strict, so a mismatch here is a loud shape error
    # rather than a model quietly reading the wrong columns.
    vec_dim = vector_dim(dcfg)
    model = ACT(n_cameras=n_cameras, proprio_dim=vec_dim, action_dim=vec_dim,
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


def _live_head_proprio(pan: float, tilt: float) -> np.ndarray:
    """pan(1) + tilt(1) -- matches dataset.yaml's head.proprio, which is
    observations/head/q, i.e. the MEASURED neck angles.

    The values come off SensorFrame, which LiveSource fills from the head
    channel when it receives and from SceneObjectsMsg otherwise. Both carry the
    same two joints in the same order and the same units (radians) that
    head.csv's q_0/q_1 were logged in, so this is a straight read with no
    conversion. If those ever diverge it will not look like an error -- the
    policy will simply act as though the camera points somewhere it does not.
    """
    return np.asarray([pan, tilt], dtype=np.float32)


def _slerp_wxyz(q0, q1, t: float) -> np.ndarray:
    """Shortest-arc interpolation between two (w, x, y, z) quaternions.

    Written out rather than going through scipy's Slerp, which wants a
    key-times/rotations pair and is built for interpolating a whole sequence --
    this is one pair, once per arm per tick, on the loop's critical path.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1  # same rotation, opposite hemisphere; take the short way round
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)  # nearly parallel: lerp, and sin(theta) -> 0 below
    else:
        theta = np.arccos(dot)
        out = (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return out / np.linalg.norm(out)


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
                 head: Optional[HeadChannel] = None,
                 device: Optional[str] = None, debug_dump_dir: Optional[str] = None,
                 debug_dump_every: int = 30,
                 stored_hw: Optional[dict] = None,
                 timing_every: int = 0,
                 resume_ramp_s: float = 0.5):
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
        # Kept for callers that pass it, and unused here: the head pose this
        # module needs arrives on SensorFrame, which LiveSource fills from this
        # same channel when it receives and from SceneObjectsMsg when it does
        # not. One source, resolved once per tick, read in one place.
        self._head = head
        # Whether THIS checkpoint has the head in its vectors, read from the
        # dcfg baked into it rather than from configs/ -- the same checkpoint
        # must keep running after the config moves on.
        self._head_cfg = head_cfg(self.dcfg)
        # Where the arms end and the head begins in both vectors. _vector()
        # appends the head after every arm, so the arms keep the offsets they
        # always had and only the tail is new.
        self._n_arm_dims = self.dcfg["action"]["dims_per_arm"] * len(self.dcfg["action"]["arms"])
        if self._head_cfg:
            print(f"[PolicyModule] checkpoint includes head joints "
                  f"(proprio/action dim {vector_dim(self.dcfg)}); head pose read from SensorFrame")
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
        # Per-arm authority as of the last step, for detecting handovers. None
        # until the first frame, so the first tick is never read as a change.
        self._last_authority: Optional[dict] = None
        # Ensemble disagreement for the slot just emitted; see _ensemble.
        self._last_spread = 0.0
        # Blend-in on regaining an arm; see _apply_resume_ramp. 0 disables.
        self._resume_ramp_s = float(resume_ramp_s)
        # monotonic() at which each arm was handed back, or absent if not ramping.
        self._resume_t0: dict = {}

    def reset(self) -> None:
        """Clears the rolling chunk-prediction buffer at an episode boundary."""
        self._pending.clear()
        self._t0_ns = None  # re-anchor the training-grid slot index (see _slot_index)

    def _flush_on_authority_change(self, frame) -> bool:
        """Drops the queued chunk buffer whenever any arm changes hands.

        _pending holds up to chunk_size slots of already-predicted actions --
        40 at 20 Hz, so two seconds of motion. Without this they keep being
        ensembled and sent after the operator has taken an arm: the policy
        fights the hand that just took over, and every one of those ticks is
        logged with authority HUMAN while the command on the wire came from the
        policy. That mislabels the very segments a DAgger filter cuts on, which
        is worse than the fighting.

        Flushed in BOTH directions. Coming back the other way the buffer holds
        chunks predicted from observations taken while a human was moving the
        arm, anchored to slot indices that no longer line up with anything.

        Global rather than per-arm because the buffer is global: one prediction
        covers both arms, so there is no way to evict one arm's half of it.
        Re-predicting from the current observation is correct for the arm that
        did not change hands anyway, it just costs the ensemble's warm-up.
        """
        auth = dict(getattr(frame, "authority", None) or {})
        prev = self._last_authority
        if prev is not None and auth == prev:
            return False

        # Arms handed BACK to the policy start a resume ramp. Treat "no previous
        # reading" as "not ours yet", so a run that attaches mid-session and
        # finds an arm already in POLICY still ramps in rather than opening with
        # a step.
        for side, value in auth.items():
            was = (prev or {}).get(side)
            if value == AUTHORITY_POLICY and was != AUTHORITY_POLICY:
                self._resume_t0[side] = time.monotonic()

        self._last_authority = auth
        if prev is None:
            return False

        print(f"[PolicyModule] authority changed {prev} -> {auth}; "
              f"flushing {len(self._pending)} pending slots")
        self.reset()
        return True

    def _apply_resume_ramp(self, ee_pose: dict, measured: dict) -> dict:
        """Blends each just-resumed arm from its measured pose into the policy's
        command over resume_ramp_s. Mutates ee_pose; returns {side: alpha}.

        The symmetric half of the handover. Policy -> human is a re-anchor
        (reOrigin plus RequestCaptureOrigin) and moves the arm by 0.000 mm.
        Human -> policy had nothing: the policy simply resumes commanding
        absolute world poses from wherever the operator left the arm.

        The step that produces is bounded -- safety.max_target_lead caps the
        target at 5 cm ahead of measured -- but bounded is not the same as safe
        here, for two reasons that stack. The flush leaves _pending empty, so
        the first action is a single un-ensembled network output with none of the
        temporal averaging that normally smooths it. And a policy the operator
        just had to correct is, by construction, the one most likely to want to
        go somewhere wrong.

        So: hold the measured pose and walk into the prediction, exactly as
        PlaybackModule.start_ramp_s does for the same reason ("never opens with
        a step input"). By the time alpha reaches 1 the ensemble has refilled.

        The gripper is deliberately NOT ramped. It is a binary close flag on the
        wire (run.py thresholds at 0.04 m), so an intermediate width is not a
        gentler grasp -- it is an arbitrary side of a threshold, and a half-open
        command during a handover could drop whatever is being held.
        """
        alphas = {}
        if self._resume_ramp_s <= 0.0 or not self._resume_t0:
            return alphas

        now = time.monotonic()
        for side in list(self._resume_t0):
            alpha = (now - self._resume_t0[side]) / self._resume_ramp_s
            if alpha >= 1.0:
                del self._resume_t0[side]
                continue
            alpha = max(0.0, alpha)
            alphas[side] = alpha

            state = measured.get(side)
            pose = ee_pose.get(side)
            if state is None or pose is None:
                continue
            meas_pos = np.asarray(state.position, dtype=np.float64)
            meas_quat = np.asarray(state.quaternion, dtype=np.float64)
            ee_pose[side] = np.concatenate([
                meas_pos + alpha * (np.asarray(pose[:3], dtype=np.float64) - meas_pos),
                _slerp_wxyz(meas_quat, pose[3:7], alpha),
            ])
        return alphas

    def _apply_head_ramp(self, pan: float, tilt: float,
                         meas_pan: float, meas_tilt: float, ramp: dict):
        """Blends the commanded neck angles out of the measured ones while any
        arm is still resuming. Returns (pan, tilt, alpha).

        The arms get this treatment because a step input is bad for the robot.
        The head gets it because a step input is bad for the OPERATOR: they are
        watching through this camera, and they have just let go of a correction,
        so the one moment the view must not jump is the moment it otherwise
        would. An arm snapping 3 cm is something you notice; the whole world
        snapping 20 degrees is something you feel.

        Linear on both joints, no slerp: pan and tilt are two independent
        revolute joints, not an orientation, so there is no shortest-arc
        question to get wrong.

        Alpha is shared with the arms rather than tracked separately -- the head
        changes hands with them, so a ramp of its own could only ever disagree.
        max() across the ramping sides keeps whole-body handover (where both
        sides carry the same alpha) exact, and under per-limb it follows the
        arm that resumed most recently, which is the one still moving.
        """
        if not ramp:
            return pan, tilt, 0.0
        alpha = max(ramp.values())
        return (meas_pan + alpha * (pan - meas_pan),
                meas_tilt + alpha * (tilt - meas_tilt),
                float(alpha))

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
        # Before anything else, and before the early return below: a handover
        # can happen on a tick where the cameras are also unavailable, and the
        # queued chunks would survive it.
        self._flush_on_authority_change(frame)

        left, right = self._arm_left.latest(), self._arm_right.latest()
        if left is None or right is None or any(c not in frame.images for c in self._camera_names):
            return ActionOutput(ee_pose={}, gripper={}, extras={"skipped": "arm state or camera unavailable"})

        # Head pose off the FRAME, not off HeadChannel.
        #
        # LiveSource resolves channel-then-scene-publish once per tick and puts
        # the answer here, so there is one source and one place that reads it.
        # It is exactly as fresh as the frame itself, so nothing needs guarding.
        #
        # Reading the channel directly here is what made an earlier version
        # return "skipped" on every single tick -- the head had no port of its
        # own then, so latest() was permanently None. The checkpoint loaded,
        # the sim engaged, a run log was written, and not one command went out,
        # with nothing on screen saying why. The channel has a port again now
        # (transmission_absolute), which makes the direct read tempting and
        # still wrong: one resolved source beats two that can disagree.
        head_pan = float(getattr(frame, "head_pan", 0.0))
        head_tilt = float(getattr(frame, "head_tilt", 0.0))

        t_start = time.perf_counter()
        images = np.stack([_preprocess_image(frame.images[c], self._hw, self._stored_hw.get(c))
                           for c in self._camera_names])
        self._maybe_dump_debug_images(images)
        images_t = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)

        # Order matters and is not free to choose: it has to match _vector()'s
        # arms-then-head layout exactly, because stats_head.npz's per-dim mean
        # and std are indexed by position. A transposed pair here would
        # normalize the neck with the gripper's statistics and still run.
        proprio_parts = [_live_proprio(left), _live_proprio(right)]
        if self._head_cfg:
            proprio_parts.append(_live_head_proprio(head_pan, head_tilt))
        proprio = np.concatenate(proprio_parts)
        proprio_n = normalize(proprio, self.stats["proprio_mean"], self.stats["proprio_std"])
        proprio_t = torch.from_numpy(proprio_n).unsqueeze(0).float().to(self.device)
        t_pre = time.perf_counter()

        with torch.no_grad():
            a_hat, _, _ = self.model(images_t, proprio_t)  # z=0 at inference (actions=None)
        a_hat = denormalize(a_hat[0].cpu().numpy(), self.stats["action_mean"], self.stats["action_std"])
        t_fwd = time.perf_counter()

        action = self._ensemble(self._slot_index(frame.timestamp_ns), a_hat)
        t_end = time.perf_counter()
        self._record_timing(t_start, t_pre, t_fwd, t_end)

        d = self._d
        left_action, right_action = action[:d], action[d:2 * d]
        ee_pose = {
            "arm_left": np.concatenate([left_action[:3], _rot6d_to_quat_wxyz(left_action[3:9])]),
            "arm_right": np.concatenate([right_action[:3], _rot6d_to_quat_wxyz(right_action[3:9])]),
        }
        # After the ensemble, before anything leaves: the ramp is about what goes
        # on the wire, not about what the model thinks.
        ramp = self._apply_resume_ramp(ee_pose, {"arm_left": left, "arm_right": right})

        # Head, taken from the ENSEMBLED vector rather than a_hat[0]. _ensemble
        # averages every column with the same weights, so the neck gets the
        # identical temporal smoothing the arms get, for free. Reading the raw
        # chunk here would give a camera that jitters against arms that do not.
        head_extras = {}
        if self._head_cfg:
            h = self._n_arm_dims
            pan, tilt = float(action[h]), float(action[h + 1])
            pan, tilt, head_alpha = self._apply_head_ramp(pan, tilt, head_pan, head_tilt, ramp)
            head_extras = {"head_pan": pan, "head_tilt": tilt, "resume_ramp_head": head_alpha}

        return ActionOutput(
            ee_pose=ee_pose,
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
                # Both ride along as scalar extras, which RunLogger already
                # writes as their own series. ensemble_spread is the
                # disagreement described in _ensemble; pending_slots is how much
                # queued motion a handover would have to flush, which is the
                # thing to look at first if an intervention feels like it
                # fought back.
                "ensemble_spread": self._last_spread,
                "pending_slots": float(len(self._pending)),
                # Preprocess + forward + ensemble for this tick, in ms. Logged as
                # its own series and shown as INF on the HUD.
                "inference_ms": (t_end - t_start) * 1e3,
                # 0 = not ramping (the steady state, and every autonomous run).
                # Non-zero marks the ticks just after a handover, which is where
                # to look first if an arm behaved oddly on resume.
                "resume_ramp_left": float(ramp.get("arm_left", 0.0)),
                "resume_ramp_right": float(ramp.get("arm_right", 0.0)),
                # head_pan/head_tilt are what run.py's on_tick forwards to
                # head.send_command, in place of the fixed look-down offset it
                # falls back to for a checkpoint without a head.
                **head_extras,
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

        # Spread across the chunks that all predicted this slot, from different
        # observations at different times. It is free here (preds is already
        # stacked) and it is the closest thing to a confidence signal this setup
        # has: the CVAE runs with z=0 at inference, so the model emits no
        # uncertainty of its own. High spread means the successive observations
        # disagreed about what to do next, which is where an operator is most
        # likely to want to intervene.
        #
        # Reported, never acted on. Nothing gates or blends on it -- this is an
        # intervention system, not shared control, and a number that silently
        # moved the arm would make it one.
        self._last_spread = float(preds.std(axis=0).mean()) if len(preds) > 1 else 0.0

        ages = np.arange(len(preds))[::-1]  # 0 = most recent prediction
        w = np.exp(-M_ENSEMBLE * ages)
        w /= w.sum()
        return (w[:, None] * preds).sum(axis=0)
