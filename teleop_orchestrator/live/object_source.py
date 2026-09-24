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
from dataclasses import dataclass, field
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


# CommandAuthority (teleop-simulator/include/common.hpp): which channel the
# avatar is currently letting move an arm. Per arm, because the operator can be
# correcting one hand while the policy keeps driving the other.
AUTHORITY_POLICY = 0
AUTHORITY_HUMAN = 1
AUTHORITY_HOLD = 2
# Nobody has claimed that arm, so the avatar is gating nothing -- the behaviour
# that existed before authority. This is also what an avatar too old to publish
# the field looks like, which is why it is the default everywhere rather than
# HOLD: defaulting to HOLD would silently stop every autonomous run against an
# un-upgraded sim, with no error to explain it.
AUTHORITY_UNSET = 255


@dataclass
class ObjectFrame:
    """One tick's worth of candidate/bin slots, with the frame_id LiveSource
    joins gaze and proprio against."""
    frame_id: int
    timestamp_ns: int
    slots: list
    # {"arm_left": int, "arm_right": int}, CommandAuthority per arm. The avatar
    # re-sends this every tick rather than publishing an event, so a dropped
    # datagram costs 10 ms of staleness instead of leaving the policy acting on
    # an authority that was revoked. Empty = the avatar did not say.
    authority: dict = field(default_factory=dict)
    # Avatar SysState, carried here so it can be read without the reliable
    # command channel. That channel is point-to-point on the avatar side, so
    # with the VR interface also connected it can only serve one client; this
    # socket has its own host/port and no such contention. 255 = UNDEFINED,
    # which is also what an avatar predating the field looks like.
    state: int = 255
    # Head pan/tilt in radians, carried for the same reason as `state`: the
    # head's own channel is point-to-point and the VR interface needs it, so
    # anything that only wants the head POSE takes it from here instead.
    head_pan: float = 0.0
    head_tilt: float = 0.0


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
            # .get, not [..]: an avatar that predates the authority field is a
            # perfectly good avatar, it just is not gating anything.
            authority = {str(k): int(v) for k, v in (msg.get("authority") or {}).items()}
            frame = ObjectFrame(frame_id=msg["frame_id"], timestamp_ns=msg["timestamp_ns"],
                                slots=slots, authority=authority,
                                state=int(msg.get("state", 255)),
                                head_pan=float(msg.get("head_pan", 0.0)),
                                head_tilt=float(msg.get("head_tilt", 0.0)))
            with self._lock:
                self._latest = frame
                self._latest_time = time.monotonic()

    def __enter__(self) -> "SimObjectSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
