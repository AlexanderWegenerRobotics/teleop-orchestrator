"""Live candidate/bin geometry -- the swap point between sim-privileged state
today and a real perception pipeline later. ObjectSource is the interface
LiveSource depends on; SimObjectSource is the first (and today, only)
implementation, reading Avatar::sendSceneObjects's SceneObjectsMsg publish
(config: avatar.scene_objects in robot_config_local.yaml). A future
VisionObjectSource would implement the same interface from detections
instead -- nothing else in the orchestrator would need to change.
"""

from __future__ import annotations

import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import msgpack

from teleop_orchestrator.contracts.features import CAND_BIN, CAND_OBJECT, CAND_UNKNOWN, SLOT_PICK_OBJ, SLOT_PLACE

_RAW_TO_CAND_TYPE = {SLOT_PICK_OBJ: CAND_OBJECT, SLOT_PLACE: CAND_BIN}


@dataclass
class ObjectSlot:
    """One candidate/bin's live world geometry; object-type-agnostic wrt
    where it came from (sim query today, vision detection later)."""
    name: str
    type: int         # CAND_OBJECT, CAND_BIN, or CAND_UNKNOWN (contracts.features)
    position: tuple    # (x, y, z) world position
    quaternion: tuple  # (w, x, y, z) world orientation
    half_extents: tuple  # (x, y, z); zeros if unknown/not a box


@dataclass
class ObjectFrame:
    """One tick's worth of candidate/bin slots, with the frame_id LiveSource
    joins gaze and proprio against."""
    frame_id: int
    timestamp_ns: int
    slots: list


class ObjectSource(ABC):
    """Live source of candidate/bin geometry, polled once per orchestrator tick."""

    @abstractmethod
    def reset(self) -> None:
        """Clears any buffered state at an episode boundary."""
        ...

    @abstractmethod
    def latest(self) -> Optional[ObjectFrame]:
        """Returns the most recent ObjectFrame, or None if nothing has arrived
        yet or the feed has gone stale -- callers must not treat None as "no
        objects in the scene", only as "no data"."""
        ...


class SimObjectSource(ObjectSource):
    """Reads the sim's SceneObjectsMsg UDP publish -- a plain msgpack struct
    (not the ReliableEnvelope-wrapped protocol gaze/state_change use), sent
    fire-and-forget once per Avatar tick alongside intention_buffer_->snapshot."""

    def __init__(self, receive_port: int, max_staleness_s: float = 1.0):
        self.max_staleness_s = max_staleness_s

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", receive_port))
        self._sock.settimeout(0.1)

        self._latest: Optional[ObjectFrame] = None
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

    def reset(self) -> None:
        """Clears the buffered frame at an episode boundary."""
        with self._lock:
            self._latest = None

    def latest(self) -> Optional[ObjectFrame]:
        with self._lock:
            frame, age = self._latest, time.monotonic() - self._latest_time
        if frame is None or age > self.max_staleness_s:
            return None
        return frame

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = msgpack.unpackb(data, raw=False)
            except Exception:
                continue
            slots = [
                ObjectSlot(
                    name=s["name"],
                    type=_RAW_TO_CAND_TYPE.get(s["type"], CAND_UNKNOWN),
                    position=tuple(s["position"]),
                    quaternion=tuple(s["quaternion"]),
                    half_extents=tuple(s["half_extents"]),
                )
                for s in msg.get("slots", [])
            ]
            frame = ObjectFrame(frame_id=msg["frame_id"], timestamp_ns=msg["timestamp_ns"], slots=slots)
            with self._lock:
                self._latest = frame
                self._latest_time = time.monotonic()

    def __enter__(self) -> "SimObjectSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
