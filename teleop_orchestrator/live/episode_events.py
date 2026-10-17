"""Receives the avatar's EpisodeEventMsg (episode_msg.hpp) so the local run can
rotate its logs on the same session_id/episode_index the avatar uses."""

from __future__ import annotations

import socket
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import msgpack


@dataclass
class EpisodeEvent:
    type: str              # "episode_start" | "episode_end"
    session_id: str
    episode_index: int
    reason: str
    log_dir: str

    @property
    def episode_dir(self) -> str:
        """Relative folder shared by every producer: <session_id>/<episode_index:03d>."""
        return f"{self.session_id}/{self.episode_index:03d}"


class EpisodeEventReceiver:
    """Background UDP listener; events are drained on the tick thread with `pop()`."""

    def __init__(self, receive_port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", receive_port))
        self._sock.settimeout(0.1)
        self._queue: deque[EpisodeEvent] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._sock.close()

    def reset(self) -> None:
        """No-op: episode events are the boundary signal itself."""

    def pop(self) -> Optional[EpisodeEvent]:
        """Returns the oldest undelivered event, or None."""
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                m = msgpack.unpackb(data, raw=False)
                ev = EpisodeEvent(str(m.get("type", "")), str(m.get("session_id", "")), int(m.get("episode_index", -1)),
                                  str(m.get("reason", "")), str(m.get("log_dir", "")))
            except Exception:
                continue
            with self._lock:
                self._queue.append(ev)
