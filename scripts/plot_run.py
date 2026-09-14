"""Offline sanity check for a saved run.hdf5 (see teleop_orchestrator/logging.py):
plots the policy's predicted world-frame EE position and gripper per arm over
the session, so you can see what it was actually converging toward (or not)
without needing video -- same idea as teleop-policy/scripts/verify_stage1.py's
plots, just against a live run instead of a training episode.

Usage:
    python scripts/plot_run.py logs/run_20260806_193000.hdf5 [out_dir]
"""

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

ARMS = ["arm_left", "arm_right"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/run.hdf5"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(path, "r") as f:
        t_ns = f["frames/timestamp_ns"][:]
        t_s = (t_ns - t_ns[0]) / 1e9

        modules = [m.decode() if isinstance(m, bytes) else m for m in f.attrs["modules"]]
        policy_name = next((m for m in modules if f"modules/{m}/arm_left/ee_pose" in f), None)
        if policy_name is None:
            print(f"no policy module with ee_pose found (modules present: {modules})")
            return

        fig, axes = plt.subplots(len(ARMS), 4, figsize=(16, 6), sharex=True)
        for row, arm in enumerate(ARMS):
            key = f"modules/{policy_name}/{arm}"
            if f"{key}/ee_pose" not in f:
                continue
            ee_pose = f[f"{key}/ee_pose"][:]     # [T, 7]: pos(3) + quat_wxyz(4)
            gripper = f[f"{key}/gripper"][:]      # [T]
            pos = ee_pose[:, :3]

            for c, label in enumerate(("x", "y", "z")):
                ax = axes[row, c]
                ax.plot(t_s, pos[:, c])
                ax.set_title(f"{arm} world pos[{label}]")
                if row == len(ARMS) - 1:
                    ax.set_xlabel("t (s)")

            ax = axes[row, 3]
            ax.plot(t_s, gripper)
            ax.set_title(f"{arm} gripper")
            if row == len(ARMS) - 1:
                ax.set_xlabel("t (s)")

            # crude "is it converging or drifting/idling" signal: per-tick
            # displacement -- near-zero for a long stretch after an initial
            # approach suggests idling, not necessarily a bug (could be a
            # correct, stable grasp/hold), but worth eyeballing against
            # the position plots above
            speed = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            print(f"{arm}: mean |dpos/tick| = {speed.mean():.5f}  "
                  f"last 1s mean = {speed[-30:].mean():.5f}  (near-zero = idling)")

        fig.tight_layout()
        out_path = os.path.join(out_dir, "run_ee_pose.png")
        fig.savefig(out_path, dpi=120)
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
