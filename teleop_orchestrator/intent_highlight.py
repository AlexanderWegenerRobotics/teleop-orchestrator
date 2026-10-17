"""Sends the intent module's top target per arm to the VR interface as a highlight (development aid)."""

from __future__ import annotations

import socket
import time

import msgpack
import numpy as np

from .contracts import IntentOutput, SensorFrame
from .contracts.features import CAND_GOAL


class IntentHighlightSender:
    """One msgpack datagram per tick (rate limited) with the candidates whose target probability clears min_prob."""

    def __init__(self, host: str, port: int, min_prob: float = 0.5, hz: float = 20.0):
        self.addr = (host, int(port))
        self.min_prob = min_prob
        self.period = 1.0 / hz
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last = 0.0

    def send(self, frame: SensorFrame, intent: IntentOutput) -> None:
        now = time.monotonic()
        if now - self._last < self.period or frame.candidate_world_pos is None:
            return
        self._last = now
        names = frame.candidate_names
        targets = []
        for side in ("left", "right"):
            post = np.asarray(intent.arm(side).target_posterior, dtype=float)
            if post.size == 0:
                continue
            i = int(np.argmax(post))
            if i >= frame.candidate_mask.size or not frame.candidate_mask[i] or post[i] < self.min_prob:
                continue
            pos = frame.candidate_world_pos[i]
            if not np.all(np.isfinite(pos)):
                continue
            name = names[i] if i < len(names) else f"cand_{i}"
            kind = "goal" if int(frame.candidate_types[i]) == CAND_GOAL else "object"
            targets.append({"name": name, "position": [float(v) for v in pos], "prob": float(post[i]), "arm": side,
                            "kind": kind})
        self.sock.sendto(msgpack.packb({"type": "highlight", "targets": targets, "t_ns": time.time_ns()}, use_bin_type=True), self.addr)

    def close(self) -> None:
        self.sock.close()
