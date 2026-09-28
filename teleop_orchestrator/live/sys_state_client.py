"""UDP client for the avatar's cmd channel (UdpReliable, see udp_reliable.hpp
and avatar.cpp's registerHandler("state_change"/"arm_reset"/"arm_resume")):
requests SysState transitions, sends the heartbeat the deadman switch needs
to hold ENGAGED, and tracks the avatar's echoed state. Same protocol
teleop-simulator/tests/arm_tester.py exercises, generalized into a reusable
class. Owns no policy about *when* to request which state -- that's
SystemArbitrator's job; this is transport only.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Optional

from . import wire

HEARTBEAT_INTERVAL_S = 0.5


class SysStateClient:
    """Speaks the avatar.transmission cmd channel: state_change / arm_reset /
    arm_resume out, heartbeat out, avatar's echoed state in."""

    def __init__(self, remote_ip: str, send_port: int, receive_port: int):
        self.remote_ip = remote_ip
        self.send_port = send_port

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", receive_port))
        self._sock.settimeout(0.1)

        self._seq = 0
        self._state = wire.SysState.OFFLINE
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._start_time = time.monotonic()

    def start(self) -> None:
        """Starts the background receive and heartbeat threads. The heartbeat
        is the avatar's deadman switch -- ENGAGED reverts to IDLE without it
        (see Avatar::start's cmd_channel_->isAlive() check) -- so it must run
        for the whole session, not just while a state change is pending."""
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._recv_thread.start()
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Requests IDLE, stops both threads, and closes the socket."""
        self.request_state(wire.SysState.IDLE)
        time.sleep(0.2)
        self._running = False
        for t in (self._recv_thread, self._heartbeat_thread):
            if t:
                t.join(timeout=1.0)
        self._sock.close()

    @property
    def state(self) -> int:
        """Avatar's last echoed SysState, from any received envelope (not just state_change acks)."""
        with self._lock:
            return self._state

    def request_state(self, requested_state: int) -> None:
        """Sends a state_change request; not synchronous -- poll .state to see it take effect."""
        self._send("state_change", {"requested_state": requested_state})

    def request_authority(self, authority: int, device: Optional[str] = None) -> None:
        """Claims or releases arms for the policy (CommandAuthority, see
        live.object_source.AUTHORITY_*). device=None addresses every arm.

        source is always "orchestrator", which the avatar arbitrates on
        (Avatar::applyAuthorityRequest): an operator claim outranks this one, so
        a POLICY request here is REFUSED while the operator holds an arm. That
        is deliberate -- the policy deciding it is ready again must never take an
        arm out from under a hand that is mid-correction. HOLD is always
        honoured, because it is the one request that cannot make anything move.

        Not synchronous. The avatar re-asserts the result every tick inside
        SceneObjectsMsg, so poll SensorFrame.authority (or the ObjectSource)
        rather than assuming this took effect.
        """
        payload = {"authority": int(authority), "source": "orchestrator"}
        if device is not None:
            payload["device"] = device
        self._send("authority_request", payload, ack_requested=True)

    def request_arm_reset(self, device: str) -> None:
        """Requests recovery for one arm ("arm_left"/"arm_right")."""
        self._send("arm_reset", {"device": device}, ack_requested=True)

    def request_arm_resume(self, device: str) -> None:
        """Confirms resume for one arm once its reset has completed (device_event/reset_complete)."""
        self._send("arm_resume", {"device": device}, ack_requested=True)

    def request_episode_restart(self, label: str = "operator_home") -> None:
        """Ends the current episode and starts a fresh one with a NEW scene.

        This is the only way to get a new scene without restarting the sim.
        Avatar::requestEpisodeConfig is called in exactly two places -- once at
        startup, and once in the episode_restart handler (avatar.cpp) -- so
        walking SysState (IDLE -> HOMING -> ENGAGED) re-homes the arms but
        leaves the objects exactly where they were. That distinction matters
        when the episode config server is in --replay mode: without this, every
        rollout faces the first replayed episode forever.

        The handler also rolls the recording folder (startNewEpisodeFolder) and
        emits episode_end/episode_start markers, so each rollout lands in its
        own directory. label is written into the episode_end event.
        """
        self._send("episode_restart", {"label": label}, ack_requested=True)

    def send_policy_status(self, inference_ms: float, agree: float) -> None:
        """Sends the HUD's policy readouts; the avatar relays them to the interface.
        Unacknowledged: a lost one is replaced by the next a fifth of a second later."""
        self._send("policy_status", {"inference_ms": float(inference_ms), "agree": float(agree)})

    def request_reset_all(self, reason: str = "reset_all") -> None:
        """Sends every arm home (recovery to q0) WITHOUT ending the episode.

        Avatar's reset_all handler starts an OPERATOR_RESET recovery on every
        arm; once all of them wait for the ack it confirms them itself
        (processResetAllCompletion) and returns to ENGAGED. It deliberately does
        not call markEpisodeEnd -- that is left to a following episode_restart,
        which is exactly the VR home button's sequence and what the evaluation
        loop uses between trials. Recovers FAULTed arms on the way.
        """
        self._send("reset_all", {"reason": reason}, ack_requested=True)

    def _send(self, msg_type: str, payload: dict, ack_requested: bool = False) -> None:
        # Locked: the heartbeat thread, the console worker and the evaluation
        # reset thread all send, and two envelopes with the same sequence number
        # could be dropped as a duplicate by the avatar's reliable channel.
        with self._send_lock:
            self._seq += 1
            packed = wire.pack_envelope(self._seq, msg_type, payload, self.state, ack_requested)
            self._sock.sendto(packed, (self.remote_ip, self.send_port))

    def _heartbeat_loop(self) -> None:
        while self._running:
            uptime_ms = int((time.monotonic() - self._start_time) * 1000)
            self._send("heartbeat", {"uptime_ms": uptime_ms})
            time.sleep(HEARTBEAT_INTERVAL_S)

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
            with self._lock:
                self._state = envelope.get("state", self._state)

    def __enter__(self) -> "SysStateClient":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
