"""Adapter between the orchestrator's SensorFrame/IntentOutput and the teleop-nudge engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from teleop_nudge import build_engine
from teleop_nudge.contracts import Arm

from .contracts import GLOBAL_FEATURE_NAMES, IntentOutput, Module, NudgeOutput, SensorFrame
from .live import wire
from .live.object_source import AUTHORITY_HUMAN, AUTHORITY_UNSET

_FEXT = {side: [GLOBAL_FEATURE_NAMES.index(f"fext_{side}_{c}") for c in ("fx", "fy", "fz", "tx", "ty", "tz")]
         for side in ("left", "right")}


def assemble(frame: SensorFrame, upstream: dict) -> tuple[SensorFrame, IntentOutput | None]:
    """Wire.assemble for the nudge module: the raw frame plus the intent output, if an intent module runs."""
    return frame, upstream.get("intent")


class NudgeModule(Module):
    """Pushes one tick of frame and intent into the engine's StateBuilder and steps it."""

    def __init__(self, nudge_cfg, log_path: str | None = None, send: bool = True, state_source: str | None = None):
        if not isinstance(nudge_cfg, dict):
            nudge_cfg = yaml.safe_load(Path(nudge_cfg).read_text())
        if state_source:
            nudge_cfg = {**nudge_cfg, "state": {**nudge_cfg.get("state", {}), "state_source": state_source}}
        self.engine = build_engine(nudge_cfg, log_path=log_path, send=send)

    def reset(self) -> None:
        self.engine.reset()

    def step(self, x: tuple[SensorFrame, IntentOutput | None]) -> NudgeOutput:
        frame, intent = x
        st = self.engine.state
        for source, arms in (frame.arm_states or {}).items():
            for name, state in arms.items():
                if state is None:
                    continue
                side = name.split("_")[1]
                st.push_robot(source, side, state.sample_time_ns or state.timestamp_ns,
                              ee_pose=np.r_[state.position, state.quaternion], q=state.joints,
                              grasp=state.grasp_state, gripper_width=state.gripper_width)
        for side in ("left", "right"):
            rs = (frame.arm_states.get("remote") or {}).get(f"arm_{side}")
            t_w = (rs.sample_time_ns or rs.timestamp_ns) if rs else frame.timestamp_ns
            st.push("wrench_" + side, frame.global_features[_FEXT[side]], t_w, "remote")
            if frame.operator is not None:
                clutched = frame.operator.clutch_left if side == "left" else frame.operator.clutch_right
                st.set_active(side, frame.engaged and not clutched)
            else:
                auth = (frame.authority or {}).get(f"arm_{side}", AUTHORITY_UNSET)
                st.set_active(side, frame.engaged and auth in (AUTHORITY_HUMAN, AUTHORITY_UNSET))
            if intent is not None:
                ai = intent.arm(side)
                st.push_intent(side, ai.top_phase(), ai.phase_posterior, ai.top_target(), ai.target_entropy())
        st.set_engaged(frame.engaged)
        snap, proposals, decision = self.engine.step(frame.timestamp_ns)
        return NudgeOutput(issued=[c.to_wire() for c in decision.issued],
                           suppressed=[(p.nudge, reason) for p, reason in decision.suppressed],
                           extras={"n_proposals": len(proposals), "state_source": snap.state_source})

    def rotate_log(self, path) -> None:
        self.engine.rotate_log(path)

    def close(self) -> None:
        self.engine.close()
