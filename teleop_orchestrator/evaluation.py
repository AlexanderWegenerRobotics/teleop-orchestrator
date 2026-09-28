"""Automated policy evaluation: N trials of a fixed length, back to back.

policy.mode: evaluate (or run.py --trials N) runs the policy exactly as
autonomous mode does, wrapped in a trial loop:

    engage (homing + the usual one-time confirmation)
    episode_restart          -> fresh scene, fresh sim log folder
    repeat N times:
        trial i: policy acts for timeout_s
        stop commanding, reset_all (arms home, FAULTs recovered)
        episode_restart(label="eval:<run_id>:<i>")  -> closes trial i's folder
                                                      with that label, loads the
                                                      next scene

The timeout is the only stopping rule. The orchestrator cannot reliably tell
that the task is done (it sees candidate positions but not bins or colours), so
every trial runs its full length and success is scored offline from the sim
logs by scripts/evaluate_policy.py.

Why reset_all before episode_restart, and not the console's next_episode
(episode_restart -> IDLE -> HOMING -> ENGAGED): that order ends the NEW
episode with operator_idle a second after it starts, and homes the arms
through a scene that has just been re-spawned. reset_all homes first, inside
the old episode, without ending it; episode_restart then closes it with the
trial's label. That is also the VR home button's sequence.

Everything one evaluation produces goes into one folder:
    logs/eval/<run_id>/manifest.json      trial boundaries, labels, config
    logs/eval/<run_id>/trial_XX.hdf5      orchestrator log of each trial
The sim writes its own episode folders; evaluate_policy.py finds each trial's
folder by its episode_end label and can copy them in (--copy-sim).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Optional

from .live import wire
from .live.object_source import AUTHORITY_HOLD, AUTHORITY_POLICY
from .logging import RunLogger


class RoutingLogger:
    """Stands in for RunLogger in Orchestrator.run and the actuator, and routes
    every record to the CURRENT trial's RunLogger. swap() hands back the
    finished one and starts a new one; ticks between trials (homing, scene
    reload) land in a logger that is simply discarded."""

    def __init__(self):
        self.current = RunLogger()
        self.total_ticks = 0

    @property
    def n_ticks(self) -> int:
        return self.total_ticks

    def record(self, frame, outputs) -> None:
        self.total_ticks += 1
        self.current.record(frame, outputs)

    def log_gripper(self, *a, **k) -> None:
        self.current.log_gripper(*a, **k)

    def log_arm_state(self, *a, **k) -> None:
        self.current.log_arm_state(*a, **k)

    def swap(self) -> RunLogger:
        old, self.current = self.current, RunLogger()
        return old


class EvalSession:
    """Trial loop state machine. tick() runs on the orchestrator's thread once
    per frame; homing and scene reloads run on a worker thread so the loop (and
    the sim's heartbeat-driven deadman) never stalls."""

    STARTING, READY, RUNNING, RESETTING, DONE, ABORTED = (
        "starting", "ready", "running", "resetting", "done", "aborted")

    def __init__(self, n_trials: int, timeout_s: float, sys_state, arms: dict,
                 logger: RoutingLogger, out_dir: str, run_id: str,
                 on_trial_start: Optional[Callable[[], None]] = None,
                 meta: Optional[dict] = None, settle_s: float = 1.5,
                 reset_timeout_s: float = 45.0):
        self.n_trials, self.timeout_s = int(n_trials), float(timeout_s)
        self.sys_state, self.arms, self.logger = sys_state, arms, logger
        self.out_dir, self.run_id = out_dir, run_id
        self.on_trial_start = on_trial_start
        self.settle_s, self.reset_timeout_s = settle_s, reset_timeout_s
        self.meta = dict(meta or {})
        self.trials: list = []
        self.idx = 0
        self._phase = self.STARTING
        self._lock = threading.Lock()
        self._mlock = threading.Lock()      # manifest is written from both threads
        self._t_start_ns = 0
        self._ticks = 0
        self._unholds, self._last_unhold = 0, 0.0
        os.makedirs(out_dir, exist_ok=True)

    # ── phase, shared between the loop and the worker thread ─────────────────
    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def _set(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    @property
    def acting(self) -> bool:
        """Whether the actuator may send this tick. False on a trial's first
        tick: the policy's output for it was computed before the reset."""
        return self.phase == self.RUNNING and self._ticks > 1

    @property
    def finished(self) -> bool:
        return self.phase in (self.DONE, self.ABORTED)

    def label(self, i) -> str:
        """episode_end label of trial i's sim folder -- how the offline scorer finds it."""
        return f"eval:{self.run_id}:{i:02d}" if isinstance(i, int) else f"eval:{self.run_id}:{i}"

    # ── lifecycle ────────────────────────────────────────────────────────────
    def begin(self) -> None:
        """Call once after engage(): closes whatever episode was open and loads
        the first evaluation scene, then trial 1 starts on the next tick."""
        self._write_manifest()
        print(f"[eval] {self.run_id}: {self.n_trials} trials x {self.timeout_s:g} s "
              f"-> {self.out_dir}")
        threading.Thread(target=self._restart_then, args=(self.label("pre"), self.READY),
                         daemon=True).start()

    def tick(self, frame=None) -> None:
        """Advance the state machine. Call before the actuator every tick."""
        phase = self.phase
        if phase == self.READY:
            self._start_trial()
        elif phase == self.RUNNING:
            self._ticks += 1
            self._unhold(frame)
            if (time.time_ns() - self._t_start_ns) * 1e-9 >= self.timeout_s:
                self._end_trial("timeout")

    def _unhold(self, frame) -> None:
        """Re-claim arms the avatar parked in HOLD.

        Authority is UNSET in a plain autonomous session and gates nothing. But
        once the sim has seen intervention mode it stays enforced: the avatar
        parks arms in HOLD when it leaves ENGAGED, and its staleness watchdog
        does the same after a gap in commands (an inference spike, the homing
        between trials). arm_allowed then refuses every command and the trial
        would silently run out its clock with the arms frozen. With nobody
        holding an arm a POLICY request is always granted. Throttled to 1 Hz,
        and counted per trial so it shows up in the manifest."""
        auth = (getattr(frame, "authority", None) or {}) if frame is not None else {}
        if AUTHORITY_HOLD not in auth.values():
            return
        now = time.monotonic()
        if now - self._last_unhold < 1.0:
            return
        self._last_unhold = now
        self._unholds += 1
        self.sys_state.request_authority(AUTHORITY_POLICY)

    def close(self, stopped: bool) -> None:
        """Call after the run loop exits. Saves a trial cut short by 'stop'/Ctrl+C."""
        if self.phase == self.RUNNING:
            self._end_trial("stopped", restart=False)
        if stopped and not self.finished:
            self._set(self.ABORTED)
        self._write_manifest()
        done = sum(t["end_reason"] == "timeout" for t in self.trials)
        print(f"[eval] {self.run_id}: {done}/{self.n_trials} trials completed. Score with:\n"
              f"  python scripts/evaluate_policy.py {self.out_dir} --sim-logs <teleop-simulator/logs>")

    # ── trials ───────────────────────────────────────────────────────────────
    def _start_trial(self) -> None:
        self.idx += 1
        if self.on_trial_start is not None:
            self.on_trial_start()          # policy.reset(), gripper latch reset
        self.logger.swap()                 # drop ticks recorded while homing
        self._ticks = 0
        self._unholds, self._last_unhold = 0, 0.0
        self._t_start_ns = time.time_ns()
        self._set(self.RUNNING)
        print(f"[eval] trial {self.idx}/{self.n_trials} started ({self.timeout_s:g} s)")

    def _end_trial(self, reason: str, restart: bool = True) -> None:
        t_end = time.time_ns()
        log = self.logger.swap()
        path = os.path.join(self.out_dir, f"trial_{self.idx:02d}.hdf5")
        entry = {"trial": self.idx, "label": self.label(self.idx), "end_reason": reason,
                 "start_ns": self._t_start_ns, "end_ns": t_end,
                 "duration_s": round((t_end - self._t_start_ns) * 1e-9, 3),
                 "ticks": log.n_ticks, "log": os.path.basename(path), "reset_ok": None,
                 "hold_reclaims": self._unholds}
        self.trials.append(entry)
        try:
            log.save(path, meta={"run_id": self.run_id, "trial": self.idx, "label": entry["label"],
                                 "start_ns": self._t_start_ns, "end_ns": t_end,
                                 "policy_mode": "evaluate"})
        except Exception as e:  # a failed save must not stop the arms from going home
            print(f"[eval] WARNING: could not save {path}: {e!r}")
        self._write_manifest()
        print(f"[eval] trial {self.idx} ended ({reason}, {entry['duration_s']:.1f} s, "
              f"{log.n_ticks} ticks)")
        if not restart:
            return
        self._set(self.RESETTING)
        nxt = self.DONE if self.idx >= self.n_trials else self.READY
        threading.Thread(target=self._home_restart_then,
                         args=(entry, nxt), daemon=True).start()

    # ── worker-thread steps ──────────────────────────────────────────────────
    def _home_restart_then(self, entry: dict, nxt: str) -> None:
        ok = self._home()
        entry["reset_ok"] = ok
        # Closes the trial's sim folder with its label either way: the folder
        # is complete, and the scorer needs the label to find it.
        self.sys_state.request_episode_restart(label=entry["label"])
        self._write_manifest()
        if not ok:
            print("[eval] arms did not come home -- stopping the evaluation. Check the sim; "
                  "the trials so far are saved.")
            self._set(self.ABORTED)
            return
        time.sleep(self.settle_s)          # let the avatar spawn the new scene
        self._set(nxt)

    def _restart_then(self, label: str, nxt: str) -> None:
        self.sys_state.request_episode_restart(label=label)
        time.sleep(self.settle_s)
        self._set(nxt)

    def _arm_states(self) -> dict:
        out = {}
        for side, ch in self.arms.items():
            st = ch.latest()
            out[side] = None if st is None else int(st.state)
        return out

    def _home(self) -> bool:
        """reset_all, then wait until every arm has left ENGAGED (recovering)
        and come back to ENGAGED. Resent once if nothing happens, since the
        request is a single UDP datagram."""
        engaged = wire.SysState.ENGAGED
        for attempt in (1, 2):
            self.sys_state.request_reset_all(reason=f"eval:{self.run_id}")
            t0 = time.monotonic()
            left = False
            while time.monotonic() - t0 < 5.0:
                if any(s != engaged for s in self._arm_states().values()):
                    left = True
                    break
                time.sleep(0.05)
            if left:
                break
            print(f"[eval] reset_all: arms did not start homing (attempt {attempt})")
        else:
            return False
        deadline = time.monotonic() + self.reset_timeout_s
        while time.monotonic() < deadline:
            if all(s == engaged for s in self._arm_states().values()):
                return True
            time.sleep(0.1)
        print(f"[eval] arms not ENGAGED {self.reset_timeout_s:.0f} s after reset_all: "
              f"{self._arm_states()}")
        return False

    def _write_manifest(self) -> None:
        m = dict(self.meta)
        m.update({"run_id": self.run_id, "n_trials": self.n_trials, "timeout_s": self.timeout_s,
                  "phase": self.phase, "trials": self.trials})
        with self._mlock:
            tmp = os.path.join(self.out_dir, "manifest.json.tmp")
            with open(tmp, "w") as f:
                json.dump(m, f, indent=2, default=str)
            os.replace(tmp, os.path.join(self.out_dir, "manifest.json"))
