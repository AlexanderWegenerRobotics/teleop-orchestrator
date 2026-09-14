"""UDP client for one arm's UdpStream channel (see arm_control.cpp): sends
ArmCommandMsg target poses, receives ArmStateMsg proprio. Same protocol
teleop-simulator/tests/arm_tester.py exercises, generalized into a reusable
class instead of a curses tool.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Optional

from . import wire
from .wire import ArmState


class ArmChannel:
    """One arm's UDP command/state channel (robot_config_local.yaml's
    devices[].transmission block: remote_ip/send_port/receive_port)."""

    def __init__(self, name: str, device_id: int, remote_ip: str, send_port: int,
                 receive_port: int, max_staleness_s: float = 0.5,
                 absolute_send_port: Optional[int] = None):
        self.name = name
        self.device_id = device_id
        self.remote_ip = remote_ip
        self.send_port = send_port
        # arm_control.cpp's transmission_absolute.receive_port -- a second,
        # separate command port that interprets position/quaternion as an
        # absolute world-frame pose (worldAbsoluteToBase) instead of
        # send_port's delta-from-origin VR semantics. Optional: only needed
        # for policy/autonomous actuation, not by anything using send_command.
        self.absolute_send_port = absolute_send_port
        self._max_staleness_s = max_staleness_s

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", receive_port))
        self._sock.settimeout(0.1)

        self._seq = 0
        self._latest: Optional[ArmState] = None
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

    def latest(self) -> Optional[ArmState]:
        """Returns the most recently received ArmState, or None if nothing has
        arrived yet or the feed has gone stale (arm offline/link down) --
        callers must not treat None as "arm at rest", only as "no data"."""
        with self._lock:
            state, age = self._latest, time.monotonic() - self._latest_time
        if state is None or age > self._max_staleness_s:
            return None
        return state

    def send_command(self, sys_state: int, position, quaternion, gripper: float) -> None:
        """Sends one ArmCommandMsg target pose (delta-from-origin/VR
        semantics on the sim side); quaternion is (w, x, y, z)."""
        self._seq += 1
        packed = wire.pack_arm_command(sequence=self._seq, state=sys_state, device_id=self.device_id,
                                        position=position, quaternion=quaternion, gripper=gripper)
        self._sock.sendto(packed, (self.remote_ip, self.send_port))

    def send_absolute_command(self, sys_state: int, position, quaternion, gripper: float) -> None:
        """Same ArmCommandMsg, sent to absolute_send_port instead -- the sim
        interprets it as an absolute world-frame pose (worldAbsoluteToBase),
        not a delta from origin. quaternion is (w, x, y, z). Same UDP socket,
        just a different destination port -- no separate bind needed since
        we never read this channel's ArmStateMsg echo (see arm_control.cpp's
        transmission_absolute comment)."""
        if self.absolute_send_port is None:
            raise RuntimeError(f"{self.name}: absolute_send_port not configured")
        self._seq += 1
        packed = wire.pack_arm_command(sequence=self._seq, state=sys_state, device_id=self.device_id,
                                        position=position, quaternion=quaternion, gripper=gripper)
        self._sock.sendto(packed, (self.remote_ip, self.absolute_send_port))

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(wire.ARM_STATE_SIZE + 64)
            except socket.timeout:
                continue
            except OSError:
                break
            state = wire.unpack_arm_state(data)
            if state is None:
                continue
            with self._lock:
                self._latest = state
                self._latest_time = time.monotonic()

    def __enter__(self) -> "ArmChannel":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
