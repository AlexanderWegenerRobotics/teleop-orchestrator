"""UDP receiver for live gaze samples (GazeSampleMsg, see intention_sample.hpp),
sent directly to the orchestrator rather than relayed through avatar's cmd
channel -- gaze fusion/inference now happens in the Python intent models, not
avatar.cpp's IntentionBuffer (see architecture discussion: the C++ belief
filter is a superseded baseline, contracts/features.py already excludes it as
leakage). Same ReliableEnvelope wire format the sim already speaks
(msg_type "gaze_sample"), just a new destination -- the sender (a UE5/HTC
Vive Pro Eye bridge, already doing its own latency compensation before
tagging a sample with frame_id) is unaffected in payload shape.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from . import wire


@dataclass
class GazeSample:
    """One decoded GazeSampleMsg, plus local arrival time for staleness checks."""
    frame_id: int
    gaze_px_x: float
    gaze_px_y: float
    timestamp_ns: int          # operator-side capture time (already latency-compensated upstream)
    timestamp_arrival_ns: int  # orchestrator-side receive time


class GazeReceiver:
    """Listens for gaze_sample envelopes and buffers recent samples by
    frame_id, so LiveSource can join gaze to a tick the same way
    IntentionBuffer::lookup did in C++ -- exact frame_id match, no
    interpolation (gaze arrives faster than the tick rate in practice)."""

    def __init__(self, receive_port: int, max_buffer: int = 300, max_staleness_s: float = 1.0):
        self.max_staleness_s = max_staleness_s
        self._max_buffer = max_buffer

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", receive_port))
        self._sock.settimeout(0.1)

        self._buffer: "OrderedDict[int, GazeSample]" = OrderedDict()
        self._latest: Optional[GazeSample] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the background receive thread."""
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the receive thread and closes the socket."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._sock.close()

    def reset(self) -> None:
        """Clears buffered samples at an episode boundary."""
        with self._lock:
            self._buffer.clear()
            self._latest = None

    def get(self, frame_id: int) -> Optional[GazeSample]:
        """Returns the gaze sample for an exact frame_id, or None if it never
        arrived (dropped packet) or has already been evicted from the buffer."""
        with self._lock:
            return self._buffer.get(frame_id)

    def latest(self) -> Optional[GazeSample]:
        """Returns the most recently received sample, or None if nothing has
        arrived yet or the feed has gone stale (headset/bridge disconnected) --
        callers must not treat None as "no gaze intended", only as "no data"."""
        with self._lock:
            sample = self._latest
        if sample is None:
            return None
        age_s = (time.time_ns() - sample.timestamp_arrival_ns) / 1e9
        return sample if age_s <= self.max_staleness_s else None

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                envelope = wire.unpack_envelope(data)
            except Exception:
                continue
            if envelope.get("msg_type") != "gaze_sample":
                continue
            payload = envelope.get("payload", {})
            sample = GazeSample(
                frame_id=payload.get("frame_id", 0),
                gaze_px_x=payload.get("gaze_px_x", 0.0),
                gaze_px_y=payload.get("gaze_px_y", 0.0),
                timestamp_ns=payload.get("timestamp_ns", 0),
                timestamp_arrival_ns=time.time_ns(),
            )
            with self._lock:
                self._buffer[sample.frame_id] = sample
                while len(self._buffer) > self._max_buffer:
                    self._buffer.popitem(last=False)
                self._latest = sample

    def __enter__(self) -> "GazeReceiver":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
