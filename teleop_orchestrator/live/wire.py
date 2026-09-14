"""Wire formats shared by the sim's UDP channels -- MsgHeader/ArmCommandMsg/
ArmStateMsg (raw packed structs, see common.hpp) and the ReliableEnvelope
protocol (msgpack, see network/udp_reliable.hpp). One place so ArmChannel,
SysStateClient, and future gaze/object-info clients can't drift on formats
verified byte-for-byte against teleop-simulator's current common.hpp
(MsgHeader=15B, ArmCommandMsg=47B, ArmStateMsg=105B).
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Optional

import msgpack


class SysState:
    """Mirrors common.hpp's SysState enum."""
    OFFLINE = 0
    IDLE = 1
    HOMING = 2
    AWAITING = 3
    ENGAGED = 4
    PAUSED = 5
    FAULT = 6
    STOP = 7
    RECOVERING = 8
    UNDEFINED = 255

    NAMES = {0: "offline", 1: "idle", 2: "homing", 3: "awaiting", 4: "engaged",
             5: "paused", 6: "fault", 7: "stop", 8: "recovering", 255: "undefined"}


class GraspState:
    """Mirrors common.hpp's GraspState enum. ArmStateMsg's trailing byte is
    this, not a plain "grasp_confirmed" bool -- see arm_tester.py's stale
    field naming, not carried forward here."""
    OPEN = 0
    HELD = 1
    LOST = 2

    NAMES = {0: "open", 1: "held", 2: "lost"}


class DeviceId:
    """Mirrors common.hpp's DeviceId enum."""
    LEFT_ARM = 1
    RIGHT_ARM = 2
    HEAD = 3
    AVATAR = 4


class FaultCode:
    """Mirrors common.hpp's FaultCode enum; the orchestrator only ever sends NONE."""
    NONE = 0


def now_ns() -> int:
    """Current wall-clock time in nanoseconds, matching common.hpp's timestamp_ns()."""
    return time.time_ns()


# ── MsgHeader / ArmCommandMsg / ArmStateMsg -- raw packed structs ──────────

_HEADER_FMT = "<IQBBB"  # sequence, timestamp_ns, state, fault_code, device_id
HEADER_SIZE = struct.calcsize(_HEADER_FMT)

ARM_CMD_FMT = _HEADER_FMT + "3f4ff"  # + position, quaternion(w,x,y,z), gripper
ARM_CMD_SIZE = struct.calcsize(ARM_CMD_FMT)

ARM_STATE_FMT = _HEADER_FMT + "3f4f7f7fBfB"  # + position, quat, joints, tau_ext, recovering, gripper_width, grasp_state
ARM_STATE_SIZE = struct.calcsize(ARM_STATE_FMT)


def pack_arm_command(*, sequence: int, state: int, device_id: int,
                      position, quaternion, gripper: float) -> bytes:
    """Packs one ArmCommandMsg; quaternion is (w, x, y, z)."""
    px, py, pz = position
    qw, qx, qy, qz = quaternion
    return struct.pack(ARM_CMD_FMT, sequence, now_ns(), state, FaultCode.NONE, device_id,
                        px, py, pz, qw, qx, qy, qz, gripper)


HEAD_CMD_FMT = _HEADER_FMT + "2f"  # + pan, tilt
HEAD_CMD_SIZE = struct.calcsize(HEAD_CMD_FMT)

HEAD_STATE_FMT = _HEADER_FMT + "2f"  # + pan, tilt
HEAD_STATE_SIZE = struct.calcsize(HEAD_STATE_FMT)


def pack_head_command(*, sequence: int, state: int, pan: float, tilt: float) -> bytes:
    """Packs one HeadCommandMsg."""
    return struct.pack(HEAD_CMD_FMT, sequence, now_ns(), state, FaultCode.NONE, DeviceId.HEAD, pan, tilt)


@dataclass
class HeadState:
    """One decoded HeadStateMsg."""
    sequence: int
    timestamp_ns: int
    state: int
    fault_code: int
    device_id: int
    pan: float
    tilt: float


def unpack_head_state(data: bytes) -> Optional[HeadState]:
    """Decodes one HeadStateMsg, or None if the packet is the wrong size (drop, not crash)."""
    if len(data) != HEAD_STATE_SIZE:
        return None
    f = struct.unpack(HEAD_STATE_FMT, data)
    return HeadState(sequence=f[0], timestamp_ns=f[1], state=f[2], fault_code=f[3], device_id=f[4],
                      pan=f[5], tilt=f[6])


@dataclass
class ArmState:
    """One decoded ArmStateMsg."""
    sequence: int
    timestamp_ns: int
    state: int
    fault_code: int
    device_id: int
    position: tuple
    quaternion: tuple
    joints: tuple
    tau_ext: tuple
    recovering: bool
    gripper_width: float
    grasp_state: int


def unpack_arm_state(data: bytes) -> Optional[ArmState]:
    """Decodes one ArmStateMsg, or None if the packet is the wrong size (drop, not crash)."""
    if len(data) != ARM_STATE_SIZE:
        return None
    f = struct.unpack(ARM_STATE_FMT, data)
    return ArmState(
        sequence=f[0], timestamp_ns=f[1], state=f[2], fault_code=f[3], device_id=f[4],
        position=f[5:8], quaternion=f[8:12], joints=f[12:19], tau_ext=f[19:26],
        recovering=bool(f[26]), gripper_width=f[27], grasp_state=f[28],
    )


# ── ReliableEnvelope -- msgpack, see network/udp_reliable.hpp ──────────────

def pack_envelope(sequence: int, msg_type: str, payload: dict, state: int,
                   ack_requested: bool = False) -> bytes:
    """Packs one ReliableEnvelope (sequence, timestamp_ns, state, fault_code,
    msg_type, ack_requested, payload). Caller owns the sequence counter."""
    envelope = {
        "sequence": sequence,
        "timestamp_ns": now_ns(),
        "state": state,
        "fault_code": FaultCode.NONE,
        "msg_type": msg_type,
        "ack_requested": ack_requested,
        "payload": payload,
    }
    return msgpack.packb(envelope, use_bin_type=True)


def unpack_envelope(data: bytes) -> dict:
    """Decodes one ReliableEnvelope into a plain dict."""
    return msgpack.unpackb(data, raw=False)
