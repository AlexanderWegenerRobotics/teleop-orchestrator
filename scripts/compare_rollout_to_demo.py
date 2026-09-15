"""Score a sim policy rollout against the demonstration of the SAME scene.

This is the second half of the closed-loop-in-sim harness. The first half is
teleop-simulator/scripts/episode_config_server.py --replay, which serves a
recorded episode's exact scene (object poses, spawn yaw, scale, lighting, seed,
mode) to the Avatar so the policy faces what the demonstrator faced. This side
reads the resulting run log and answers, per arm, "how did it differ".

Why it exists: teleop-policy/scripts/check_closed_loop.py closes the PROPRIO
loop but replays logged images, so the policy always sees a correctly-executing
demonstrator rather than its own mistakes. That test says the model descends
past the demo's grasp depth (negative shortfall, 3/3 episodes), while the live
robot stops ~34 mm high. The difference between those two results is the vision
loop, and only a rendered rollout can close it. This script measures the same
quantities on both sides of that gap so the numbers are directly comparable.

Metrics, per arm:
  reach depth      lowest commanded z, against the demo's commanded z floor.
                   The demos press against a hard clamp at that floor on every
                   grasp, so it is a sharp target rather than a soft average.
  closest approach EE-to-nearest-object distance, split into horizontal and
                   vertical. The split matters: a lateral miss and a failure to
                   descend look identical in the 3D norm but need different
                   fixes. (Live so far: horizontally fine, ~30 mm high.)
  grasp            attempts, close commands actually issued, confirmations.

Note EE-to-object is measured to the object CENTRE from the EE frame, which
sits ~100 mm back from the fingertips -- demos grasp at 99.4 +/- 9 mm, not at 0.
The demo column is computed the same way, so compare the two, not the absolute.

Usage:
    python scripts/compare_rollout_to_demo.py logs/run_XXXX.hdf5 \
        --episode 014 --store-root C:/Users/ceti/Documents/05_data/teleop-sorting/avatar
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

ARMS = ["arm_left", "arm_right"]
ENGAGED_STATE = 4
CAND_TYPE_OBJECT = 0
CLOSED_WIDTH_M = 0.04  # gripper_cmd is binary 0.0 / 0.08; midpoint splits it


def demo_reference(store_root: str, episode: str):
    """Per-arm reference numbers from the demonstration of this same scene."""
    path = os.path.join(store_root, str(episode).zfill(3), "episode.hdf5")
    out = {}
    with h5py.File(path, "r") as f:
        engaged = np.ones(f[f"observations/{ARMS[0]}/state"].shape[0], dtype=bool)
        for arm in ARMS:
            engaged &= f[f"observations/{arm}/state"][:] == ENGAGED_STATE
        idx = np.where(engaged)[0]
        sl = slice(int(idx.min()), int(idx.max()))

        g = f["observations/intent"]
        obj_slots = []
        for i in range(32):
            if f"slot_pos_x_{i}" not in g:
                continue
            nm = g[f"slot_name_{i}"][0]
            nm = nm.decode() if isinstance(nm, bytes) else str(nm)
            if "bin" in nm.lower():
                continue
            obj_slots.append(i)
        P = (np.stack([np.stack([g[f"slot_pos_{c}_{i}"][sl] for c in "xyz"], 1)
                       for i in obj_slots], 1) if obj_slots else None)

        for arm in ARMS:
            cmd = f[f"actions/{arm}/O_T_EE_cmd_world"][sl]
            grip = f[f"actions/{arm}/gripper_cmd"][sl]
            ee = f[f"observations/{arm}/O_T_EE_world"][sl][:, 12:15]
            rec = {"z_floor": float(cmd[:, 14].min()),
                   "n_grasps": int((np.diff((grip < CLOSED_WIDTH_M).astype(int)) > 0).sum())}
            if P is not None:
                D = P - ee[:, None, :]
                n = np.linalg.norm(D, axis=2)
                n = np.where(np.isfinite(n), n, np.inf)
                k = np.argmin(n, 1)
                closes = np.where(np.diff((grip < CLOSED_WIDTH_M).astype(int)) > 0)[0]
                v = np.stack([D[c, k[c]] for c in closes]) if len(closes) else np.zeros((0, 3))
                v = v[np.isfinite(v).all(1)]
                if len(v):
                    rec["grasp_horiz"] = float(np.median(np.hypot(v[:, 0], v[:, 1])))
                    rec["grasp_vert"] = float(np.median(v[:, 2]))
                rec["closest"] = float(np.nanmin(n[np.isfinite(n).any(1)].min(1)))
            out[arm] = rec
    return out


def _action_module(f) -> str:
    """Which module drove the arms in this run: 'policy', or 'playback' for a
    recorded-trajectory run (see teleop_orchestrator/playback_module.py)."""
    for name in ("policy", "playback"):
        if f"modules/{name}" in f:
            return name
    raise KeyError("run log has no policy or playback module output")


def rollout_metrics(path: str):
    """Same quantities, measured on a live/sim run log."""
    out = {}
    with h5py.File(path, "r") as f:
        mod = _action_module(f)
        has_cmd = "commanded" in f
        has_cand = "frames/candidate_world_pos" in f
        if has_cand:
            cw = f["frames/candidate_world_pos"][:]
            sel = f["frames/candidate_mask"][:] & (f["frames/candidate_types"][:] == CAND_TYPE_OBJECT)

        for arm in ARMS:
            pred = f[f"modules/{mod}/{arm}/gripper"][:]
            rec = {
                "z_cmd_min": float(f[f"modules/{mod}/{arm}/ee_pose"][:][:, 2].min()),
                "grip_pred_min": float(pred.min()),
                "n_attempts": int((np.diff((pred < np.percentile(pred, 90) - 0.005).astype(int)) > 0).sum()),
            }
            if has_cmd:
                flag = f[f"commanded/{arm}/gripper_close_flag"][:]
                grasp = f[f"commanded/{arm}/meas_grasp_state"][:]
                rec["n_close_cmds"] = int((np.diff((flag > 0.5).astype(int)) > 0).sum())
                rec["n_confirmed"] = int((np.diff((np.nan_to_num(grasp) > 1).astype(int)) > 0).sum())
                if has_cand:
                    ee = f[f"commanded/{arm}/meas_ee_pos"][:]
                    D = cw - ee[:, None, :]
                    n = np.linalg.norm(D, axis=2)
                    n = np.where(sel & np.isfinite(n), n, np.inf)
                    k = np.argmin(n, 1)
                    d = n[np.arange(len(n)), k]
                    if np.isfinite(d).any():
                        j = int(np.nanargmin(d))
                        v = D[j, k[j]]
                        rec["closest"] = float(d[j])
                        rec["closest_horiz"] = float(np.hypot(v[0], v[1]))
                        rec["closest_vert"] = float(v[2])
            out[arm] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--episode", required=True, help="the episode id that was replayed into the sim")
    ap.add_argument("--store-root", default="C:/Users/ceti/Documents/05_data/teleop-sorting/avatar")
    args = ap.parse_args()

    demo = demo_reference(args.store_root, args.episode)
    live = rollout_metrics(args.run)

    print(f"\nrollout : {args.run}")
    print(f"demo    : episode {str(args.episode).zfill(3)}\n")

    for arm in ARMS:
        d, l = demo[arm], live[arm]
        print(f"=== {arm}")
        short = (l["z_cmd_min"] - d["z_floor"]) * 1000
        verdict = "REACHES" if short < 5 else "SHORT"
        print(f"  reach depth      commanded z min {l['z_cmd_min']:.4f} vs demo floor "
              f"{d['z_floor']:.4f}  -> {short:+.1f} mm  [{verdict}]")

        if "closest" in l and "closest" in d:
            print(f"  closest approach {l['closest'] * 1000:6.1f} mm "
                  f"(horiz {l['closest_horiz'] * 1000:5.1f}, vert {l['closest_vert'] * 1000:+6.1f})"
                  f"   demo closest {d['closest'] * 1000:.1f} mm")
        if "grasp_horiz" in d:
            print(f"  demo grasps at   horiz {d['grasp_horiz'] * 1000:5.1f} mm, "
                  f"vert {d['grasp_vert'] * 1000:+6.1f} mm")
            if "closest_vert" in l:
                gap = (l["closest_vert"] - d["grasp_vert"]) * -1000
                print(f"  vertical gap     {gap:+.1f} mm "
                      f"({'hovering high' if gap > 10 else 'in range'})")

        print(f"  gripper          predicted min {l['grip_pred_min']:.4f} "
              f"(threshold {CLOSED_WIDTH_M}), {l['n_attempts']} attempt(s)")
        if "n_close_cmds" in l:
            print(f"                   {l['n_close_cmds']} close command(s) issued, "
                  f"{l['n_confirmed']} confirmed   |   demo made {d['n_grasps']} grasp(s)")
        print()

    print("Reading: if reach depth says REACHES here but the model still fails, the")
    print("problem is downstream of the descent. If it says SHORT here while")
    print("check_closed_loop.py reports a negative shortfall on the same episode,")
    print("the gap is the vision loop -- the model destabilises once it is looking")
    print("at its own arms rather than the demonstrator's.")


if __name__ == "__main__":
    main()
