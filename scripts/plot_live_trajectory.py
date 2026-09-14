"""Full-episode trajectory plot for a live autonomous run, with a training
episode overlaid as the "what converging looks like" reference.

Motivation: every diagnostic in act_policy_debug_handoff.md sampled the policy
at a single tick (t0, then t0+7.5s). A single tick can't distinguish "stalled"
from "slow" -- you need the whole time series. This plots it.

What it shows, per arm:
  row 1  commanded EE position per axis vs wall-clock time (live vs reference)
  row 2  per-tick displacement magnitude (mm) -- the convergence/stall signal
  row 3  gripper command -- the pick-then-place signal

Live logs (teleop_orchestrator.logging) record the *policy module's own output*
(modules/policy/<arm>/ee_pose, already temporally ensembled), NOT the measured
arm state -- so this plots what the policy asked for, which is exactly what we
want when the question is "did the policy stall or did the controller drop it".

Usage:
    python scripts/plot_live_trajectory.py logs/run_YYYYMMDD_HHMMSS.hdf5 \
        --reference-episode 014 \
        --out results/live_trajectory.png
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["arm_left", "arm_right"]
AXES = ["x", "y", "z"]
ENGAGED_STATE = 4  # matches teleop-policy configs/dataset.yaml state.train_states
GRIPPER_CLOSE_THRESHOLD_M = 0.04  # keep in sync with run.py's _GRIPPER_CLOSE_THRESHOLD_M


def load_live(path: str):
    """Live run -> dict per arm of {t, pos[n,3], grip[n]} plus frame_id.

    Also picks up the optional commanded/ group (RunLogger.log_gripper /
    log_arm_state), which records what actually went on the wire and what the
    arm actually did -- absent from runs logged before that existed, so every
    consumer of these keys must tolerate them missing.
    """
    with h5py.File(path, "r") as f:
        ts = f["frames/timestamp_ns"][:]
        t = (ts - ts[0]) / 1e9
        frame_id = f["frames/frame_id"][:]
        cmd = f["commanded"] if "commanded" in f else None
        out = {}
        for arm in ARMS:
            d = {
                "t": t,
                "pos": f[f"modules/policy/{arm}/ee_pose"][:][:, :3],
                "grip": f[f"modules/policy/{arm}/gripper"][:],
            }
            for key in ("gripper_close_flag", "meas_gripper_width", "meas_grasp_state"):
                d[key] = cmd[f"{arm}/{key}"][:] if cmd is not None and f"{arm}/{key}" in cmd else None
            out[arm] = d
    return out, t, frame_id


def load_reference(store_root: str, episode: str, rate_hz: float = 30.0):
    """Training episode -> the *commanded* world-frame EE pos + gripper over
    its ENGAGED window, i.e. the same quantity the live log stores. Column-major
    flat16, translation is elements 12:15 (see teleop-policy dataset/transforms.py)."""
    path = os.path.join(store_root, episode.zfill(3), "episode.hdf5")
    with h5py.File(path, "r") as f:
        engaged = np.ones(f[f"observations/{ARMS[0]}/state"].shape[0], dtype=bool)
        for arm in ARMS:
            engaged &= f[f"observations/{arm}/state"][:] == ENGAGED_STATE
        idx = np.where(engaged)[0]
        out = {}
        for arm in ARMS:
            flat = f[f"actions/{arm}/O_T_EE_cmd_world"][:][idx]
            out[arm] = {
                "t": (idx - idx[0]) / rate_hz,
                "pos": flat[:, 12:15],
                "grip": f[f"actions/{arm}/gripper_cmd"][:][idx],
            }
    return out


def speed_mm_s(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Commanded EE speed in mm/s, padded to len(pos).

    Deliberately per-SECOND, not per-tick: the live loop ticks at ~9 Hz while
    the demo is on a 30 Hz grid, so mm/tick would make the demo look ~3x slower
    than it is and the live/demo comparison would be meaningless.
    """
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0
    dt = np.diff(t)
    return np.concatenate([[np.nan], d / np.maximum(dt, 1e-9)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="path to a live run .hdf5")
    ap.add_argument("--store-root", default="C:/Users/ceti/Documents/05_data/teleop-sorting/avatar")
    ap.add_argument("--reference-episode", default=None,
                    help="training episode id to overlay, e.g. 014. Omit to skip.")
    ap.add_argument("--rate-hz", type=float, default=30.0,
                    help="training grid rate, for the reference episode's time axis")
    ap.add_argument("--smooth", type=int, default=5,
                    help="boxcar window (ticks) for the per-tick displacement trace")
    ap.add_argument("--out", default="results/live_trajectory.png")
    args = ap.parse_args()

    live, t, frame_id = load_live(args.run)
    ref = load_reference(args.store_root, args.reference_episode, args.rate_hz) \
        if args.reference_episode else None

    # Rate sanity: the ensembler in policy_module.py keys its pending-chunk
    # buffer by frame_id, so frame_id must advance on the SAME grid the action
    # chunk was trained on (dataset.yaml alignment.rate_hz) or the chunk is
    # replayed time-dilated. Surface both rates rather than assuming.
    src_hz = (frame_id[-1] - frame_id[0]) / t[-1]
    tick_hz = (len(t) - 1) / t[-1]
    print(f"[rate] policy ticks {tick_hz:6.2f} Hz | frame_id advances {src_hz:6.2f} Hz "
          f"| training grid {args.rate_hz:6.2f} Hz")
    if abs(src_hz - args.rate_hz) / args.rate_hz > 0.05:
        print(f"[rate] WARNING: frame_id grid is {args.rate_hz / src_hz:.2f}x slower than the "
              f"training grid -- action chunks are being replayed time-dilated.")

    fig, axarr = plt.subplots(3, 2, figsize=(15, 11), sharex="col")
    for col, arm in enumerate(ARMS):
        L = live[arm]

        ax = axarr[0, col]
        for j, name in enumerate(AXES):
            ax.plot(L["t"], L["pos"][:, j], lw=1.8, label=f"{name} (live)")
            if ref:
                ax.plot(ref[arm]["t"], ref[arm]["pos"][:, j], lw=1.0, ls="--", alpha=0.5,
                        color=ax.lines[-1].get_color(), label=f"{name} (demo)")
        ax.set_title(f"{arm} — commanded EE position")
        ax.set_ylabel("world position (m)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        ax = axarr[1, col]
        k = max(1, args.smooth)
        box = np.ones(k) / k
        s = speed_mm_s(L["pos"], L["t"])
        ax.plot(L["t"], np.convolve(np.nan_to_num(s), box, mode="same"), lw=1.8, label="live")
        if ref:
            sr = speed_mm_s(ref[arm]["pos"], ref[arm]["t"])
            ax.plot(ref[arm]["t"], np.convolve(np.nan_to_num(sr), np.ones(k * 3) / (k * 3), mode="same"),
                    lw=1.0, ls="--", alpha=0.6, label="demo")
        ax.set_yscale("log")
        ax.set_title("commanded EE speed (log) — decaying to a floor ⇒ stalled")
        ax.set_ylabel("mm / s")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        # Three separate things, deliberately on one axis: what the policy
        # wanted (predicted width), what was transmitted (close flag), and what
        # the hardware did (measured width). A gap between the first two is a
        # wiring bug; a gap between the second two is a control/actuation bug.
        ax = axarr[2, col]
        ax.plot(L["t"], L["grip"], lw=1.8, label="policy predicted width")
        if ref:
            ax.plot(ref[arm]["t"], ref[arm]["grip"], lw=1.0, ls="--", alpha=0.6, label="demo")
        if L["meas_gripper_width"] is not None:
            ax.plot(L["t"][:len(L["meas_gripper_width"])], L["meas_gripper_width"],
                    lw=1.4, alpha=0.8, label="measured width")
        ax.axhline(GRIPPER_CLOSE_THRESHOLD_M, color="grey", ls=":", lw=1.0,
                   label=f"close threshold ({GRIPPER_CLOSE_THRESHOLD_M} m)")
        ax.set_ylabel("gripper (m)")
        ax.set_xlabel("time since run start (s)")
        ax.grid(alpha=0.3)

        if L["gripper_close_flag"] is not None:
            ax2 = ax.twinx()
            flag = L["gripper_close_flag"]
            ax2.fill_between(L["t"][:len(flag)], 0, flag, step="post", alpha=0.18,
                             color="tab:red", label="close commanded")
            if L["meas_grasp_state"] is not None:
                gs = np.asarray(L["meas_grasp_state"], dtype=float)
                ax2.fill_between(L["t"][:len(gs)], 0, (gs > 1).astype(float), step="post",
                                 alpha=0.18, color="tab:green", label="grasp confirmed")
            ax2.set_ylim(0, 1.05)
            ax2.set_yticks([])
            ax2.legend(fontsize=7, loc="upper right")
            n_close = int(np.sum(np.diff((flag > 0.5).astype(int)) > 0))
            ax.set_title(f"gripper — {n_close} close command(s) issued")
        else:
            ax.set_title(f"gripper command  (live range {np.ptp(L['grip']) * 1000:.1f} mm) "
                         f"— no commanded/ group, run predates wire logging")
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(f"{os.path.basename(args.run)} — policy output over the full episode "
                 f"(ticks {tick_hz:.1f} Hz, frame_id {src_hz:.1f} Hz, trained on {args.rate_hz:.0f} Hz)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[out] wrote {args.out}")

    for arm in ARMS:
        L = live[arm]
        s = speed_mm_s(L["pos"], L["t"])
        n = len(s)
        first, last = np.nanmean(s[1:n // 4]), np.nanmean(s[-n // 4:])
        demo_note = ""
        if ref:
            sr = speed_mm_s(ref[arm]["pos"], ref[arm]["t"])
            demo_note = f" | demo mean {np.nanmean(sr):6.1f} mm/s"
        print(f"[{arm}] speed first quarter {first:6.1f} mm/s -> "
              f"last quarter {last:6.1f} mm/s  ({first / max(last, 1e-9):5.1f}x decay){demo_note} | "
              f"net {np.linalg.norm(L['pos'][-1] - L['pos'][0]) * 1000:6.1f} mm | "
              f"gripper travel {np.ptp(L['grip']) * 1000:.2f} mm")


if __name__ == "__main__":
    main()
