"""Wire formats shared by the sim's UDP channels -- MsgHeader/ArmCommandMsg/
ArmStateMsg (raw packed structs, see common.hpp) and the ReliableEnvelope
protocol (msgpack, see network/udp_reliable.hpp). One place so ArmChannel,
SysStateClient, and future gaze/object-info clients can't drift on formats
verified byte-for-byte against teleop-simulator's current common.hpp
(MsgHeader=23B, ArmCommandMsg=55B, ArmStateMsg=117B, Head*Msg=31B).

These are #pragma pack(1) C structs, so a field added on the C++ side silently
breaks BOTH directions here: inbound packets fail the size check and are
dropped (every latest() returns None), and outbound packets are the wrong
length for the receiver. Two fields were added and missed that way --
MsgHeader.sample_time_ns and ArmStateMsg.applied_cmd_sequence -- which left
the arms unable to receive a command at all while the msgpack channels
(SysStateClient, SimObjectSource) kept working, so the system looked healthy.
The size asserts at the bottom of this module exist to make the next such
change fail loudly at import.
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

# sequence, timestamp_ns, sample_time_ns, state, fault_code, device_id.
# sample_time_ns is when the DATA was sampled, timestamp_ns when the packet was
# sent; consumers measure staleness as (now - sample_time_ns).
_HEADER_FMT = "<IQQBBB"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)

ARM_CMD_FMT = _HEADER_FMT + "3f4ff"  # + position, quaternion(w,x,y,z), gripper
ARM_CMD_SIZE = struct.calcsize(ARM_CMD_FMT)

# + position, quat, joints, tau_ext, recovering, gripper_width, grasp_state,
# applied_cmd_sequence (the ArmCommandMsg sequence this arm actually consumed)
ARM_STATE_FMT = _HEADER_FMT + "3f4f7f7fBfBI"
ARM_STATE_SIZE = struct.calcsize(ARM_STATE_FMT)


def pack_arm_command(*, sequence: int, state: int, device_id: int,
                      position, quaternion, gripper: float,
                      sample_time_ns: Optional[int] = None) -> bytes:
    """Packs one ArmCommandMsg; quaternion is (w, x, y, z). sample_time_ns
    defaults to now: a command is produced at the instant it is sent."""
    px, py, pz = position
    qw, qx, qy, qz = quaternion
    now = now_ns()
    return struct.pack(ARM_CMD_FMT, sequence, now, sample_time_ns if sample_time_ns is not None else now,
                        state, FaultCode.NONE, device_id,
                        px, py, pz, qw, qx, qy, qz, gripper)


HEAD_CMD_FMT = _HEADER_FMT + "2f"  # + pan, tilt
HEAD_CMD_SIZE = struct.calcsize(HEAD_CMD_FMT)

HEAD_STATE_FMT = _HEADER_FMT + "2f"  # + pan, tilt
HEAD_STATE_SIZE = struct.calcsize(HEAD_STATE_FMT)


def pack_head_command(*, sequence: int, state: int, pan: float, tilt: float) -> bytes:
    """Packs one HeadCommandMsg."""
    now = now_ns()
    return struct.pack(HEAD_CMD_FMT, sequence, now, now, state, FaultCode.NONE, DeviceId.HEAD, pan, tilt)


@dataclass
class HeadState:
    """One decoded HeadStateMsg."""
    sequence: int
    timestamp_ns: int
    sample_time_ns: int
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
    return HeadState(sequence=f[0], timestamp_ns=f[1], sample_time_ns=f[2], state=f[3],
                      fault_code=f[4], device_id=f[5], pan=f[6], tilt=f[7])


@dataclass
class ArmState:
    """One decoded ArmStateMsg."""
    sequence: int
    timestamp_ns: int
    sample_time_ns: int
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
    applied_cmd_sequence: int


def unpack_arm_state(data: bytes) -> Optional[ArmState]:
    """Decodes one ArmStateMsg, or None if the packet is the wrong size (drop, not crash)."""
    if len(data) != ARM_STATE_SIZE:
        return None
    f = struct.unpack(ARM_STATE_FMT, data)
    return ArmState(
        sequence=f[0], timestamp_ns=f[1], sample_time_ns=f[2], state=f[3], fault_code=f[4],
        device_id=f[5], position=f[6:9], quaternion=f[9:13], joints=f[13:20], tau_ext=f[20:27],
        recovering=bool(f[27]), gripper_width=f[28], grasp_state=f[29], applied_cmd_sequence=f[30],
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


# Sizes of the packed structs in common.hpp. A mismatch here means the C++
# structs changed; every raw-struct channel silently stops working, so fail at
# import rather than at 200 Hz in a receive thread.
assert HEADER_SIZE == 23, HEADER_SIZE
assert ARM_CMD_SIZE == 55, ARM_CMD_SIZE
assert ARM_STATE_SIZE == 117, ARM_STATE_SIZE
assert HEAD_CMD_SIZE == HEAD_STATE_SIZE == 31, (HEAD_CMD_SIZE, HEAD_STATE_SIZE)
