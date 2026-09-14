"""UDP client for the head's UdpStream channel (see head_control.cpp): sends
HeadCommandMsg pan/tilt targets, receives HeadStateMsg. Same shape as
ArmChannel; head_pan/head_tilt is what LiveSource needs to reconstruct the
camera projection (R_CH) for candidate px_u/px_v -- see camera_geometry.py.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Optional

from . import wire
from .wire import HeadState


class HeadChannel:
    """The head's UDP command/state channel (robot_config_local.yaml's
    head device transmission block: remote_ip/send_port/receive_port)."""

    def __init__(self, remote_ip: str, send_port: int, receive_port: int, max_staleness_s: float = 0.5):
        self.remote_ip = remote_ip
        self.send_port = send_port
        self._max_staleness_s = max_staleness_s

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", receive_port))
        self._sock.settimeout(0.1)

        self._seq = 0
        self._latest: Optional[HeadState] = None
        self._latest_time = 0.0
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

    def latest(self) -> Optional[HeadState]:
        """Returns the most recently received HeadState, or None if nothing
        has arrived yet or the feed has gone stale."""
        with self._lock:
            state, age = self._latest, time.monotonic() - self._latest_time
        if state is None or age > self._max_staleness_s:
            return None
        return state

    def send_command(self, sys_state: int, pan: float, tilt: float) -> None:
        """Sends one HeadCommandMsg target."""
        self._seq += 1
        packed = wire.pack_head_command(sequence=self._seq, state=sys_state, pan=pan, tilt=tilt)
        self._sock.sendto(packed, (self.remote_ip, self.send_port))

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(wire.HEAD_STATE_SIZE + 64)
            except socket.timeout:
                continue
            except OSError:
                break
            state = wire.unpack_head_state(data)
            if state is None:
                continue
            with self._lock:
                self._latest = state
                self._latest_time = time.monotonic()

    def __enter__(self) -> "HeadChannel":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
