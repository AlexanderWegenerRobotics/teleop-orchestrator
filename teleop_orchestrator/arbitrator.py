"""Owns walking the sim through SysState (see live.wire.SysState) and gates
whether policy is allowed to act autonomously. The one place allowed to
request an armed/autonomous state, so "did we mean to unlock autonomous" is
answerable by looking at one module instead of scattered checks.

State sequencing matches Avatar::updateStateMachine exactly (avatar.cpp):
IDLE -[HOMING]-> HOMING -[devices reach AWAITING]-> AWAITING -[ENGAGED]->
ENGAGED. Skipping a step (e.g. requesting ENGAGED from IDLE) is a no-op on
the sim side, not an error -- so this class polls through each intermediate
state rather than assuming a single request is enough.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from .live import wire
from .live.object_source import AUTHORITY_HOLD, AUTHORITY_POLICY, AUTHORITY_UNSET
from .live.sys_state_client import SysStateClient

# The one mode where a human is in the loop with a headset on. It is handled
# apart from the others throughout this module, because in every other mode the
# only person who could consent is sitting at this terminal, and in this one
# they are not.
INTERVENTION_MODE = "intervention"

# Modes in which something issues arm commands rather than only observing.
# "intervention" actuates exactly as "autonomous" does -- the difference is not
# what it sends but who is allowed to, and that is the avatar's per-arm
# authority rather than a mode.
#
# "evaluate" is autonomous with a trial loop around it (teleop_orchestrator/
# evaluation.py): N trials of a fixed length, arms homed and a new scene loaded
# between them. For the arbitrator it is autonomous in every respect.
EVALUATE_MODE = "evaluate"
ACTUATING_MODES = ("autonomous", "playback", "intervention", EVALUATE_MODE)


class SystemArbitrator:
    """Sequences SysState transitions via SysStateClient, gated by the
    configured policy mode. Requires an explicit confirmation callback before
    entering any mode in require_confirmation_for -- by default just
    "autonomous", so ENGAGED is never entered in that mode by accident."""

    def __init__(self, client: SysStateClient, policy_mode: str,
                 require_confirmation_for=("autonomous",),
                 confirm: Optional[Callable[[str], bool]] = None,
                 state_source: Optional[Callable[[], object]] = None):
        self.client = client
        self.policy_mode = policy_mode
        self._require_confirmation_for = set(require_confirmation_for)
        self._confirm = confirm or self._confirm_cli
        # Where to READ the avatar's SysState back. Requests always go out over
        # client; observing the result is the half that cannot.
        #
        # The avatar's reliable command channel is point-to-point, so its
        # replies go to the single peer it was configured with -- the VR
        # interface. system.yaml deliberately moved this process's
        # avatar.receive_port off 8000 to 8020 so the two could coexist, and
        # its own comment says "nothing is expected to arrive here". That is
        # accurate, and it means client.state never leaves OFFLINE.
        #
        # await_operator_handover already read state out of SceneObjectsMsg for
        # exactly this reason. engage() and disengage() did not, so every
        # autonomous run requested HOMING, then watched a value that could
        # never change and gave up 30 s later -- while the sim homed perfectly
        # well, unobserved.
        #
        # Callable returning the latest SceneObjectsMsg (or None). Leave it
        # None for a caller that genuinely does receive command-channel
        # replies; the client is then read as before.
        self._state_source = state_source

    def _observed_state(self) -> int:
        """Avatar SysState as this process can actually see it."""
        if self._state_source is None:
            return self.client.state
        frame = self._state_source()
        return wire.SysState.OFFLINE if frame is None else frame.state

    @property
    def autonomous_allowed(self) -> bool:
        """Whether a module is configured to command the arms directly this
        session. 'playback' counts: a recorded trajectory moves the arms just as
        really as a policy does, so it is gated (and confirmed) the same way.

        Session-level only. It says the config permits actuation, not that any
        particular arm may be moved right now -- ask arm_allowed for that."""
        return self.policy_mode in ACTUATING_MODES

    def arm_allowed(self, frame, side: str) -> bool:
        """Whether this tick may command one arm, combining the configured mode
        with the live authority the avatar published for that arm.

        Per arm, not per robot: during an intervention the operator holds one
        arm while the policy keeps the other, and a single boolean for both
        would either freeze the arm the policy still owns or keep commanding the
        one a hand is already on.

        UNSET means the avatar is not gating anything, which is every session
        before this feature and every session where the operator never arms it,
        so it permits actuation exactly as before. HUMAN and HOLD both refuse:
        HOLD because nothing may move, HUMAN because the operator has it.

        Note this deliberately duplicates the avatar's own gate. The avatar is
        the one that ENFORCES -- it drops what it will not act on either way.
        Checking here as well means the orchestrator stops putting packets on
        the wire it knows will be discarded, so the dropped-command counters
        stay a signal about bugs rather than about normal operation.
        """
        if not self.autonomous_allowed:
            return False
        auth = (getattr(frame, "authority", None) or {}).get(side, AUTHORITY_UNSET)
        return auth in (AUTHORITY_POLICY, AUTHORITY_UNSET)

    def claim_hold(self) -> None:
        """Parks every arm in HOLD, so nothing can command them until they are
        explicitly handed over.

        Called before ENGAGED in intervention mode. Without it there is a window
        between engage() and the operator arming the panel in which authority is
        still UNSET -- which gates nothing -- so the policy has both arms with
        nobody necessarily watching, and if the headset is not up yet, nobody
        watching at all.

        Also called on shutdown: the watchdog would reach HOLD by itself 250 ms
        after this process stops sending, but saying so explicitly makes it
        immediate and puts a line in the avatar's log that says why.
        """
        self.client.request_authority(AUTHORITY_HOLD)

    def await_operator_handover(self, snapshot_of, poll_s: float = 0.2,
                                 timeout_s: float = 0.0) -> bool:
        """Blocks until the operator has engaged the sim AND handed at least one
        arm to the policy. Returns False if interrupted or timed out.

        This replaces the terminal confirmation in intervention mode, and is a
        better gate than the one it replaces. The point of _confirm_cli is that
        a human consents before the policy moves the arms -- but that prompt is
        in a window the operator cannot see from inside a headset, and it is
        answered before the robot has even homed. Pressing RESUME is the same
        consent, given by someone looking at the robot, at the moment it
        matters.

        The arms are already parked in HOLD by claim_hold(), so "nobody ever
        presses RESUME" is a safe outcome: this waits, and nothing moves.

        snapshot_of is a zero-argument callable returning the latest ObjectFrame
        (or None). Both SysState and per-arm authority are read from it rather
        than from the command channel, because that channel is point-to-point on
        the avatar side: with the VR interface connected it serves the interface,
        and self.client.state here would never advance past IDLE no matter what
        the operator did.

        Says what it is waiting for, every few seconds, forever. The first
        version of this printed once and then sat silent, which is
        indistinguishable from a hung process -- and "the terminal is frozen, I
        killed it" is what actually happened. Each line names the thing that is
        missing, so the reason is on screen rather than inferable.

        timeout_s <= 0 waits indefinitely (Ctrl+C still works; the sleep below
        is interruptible).
        """
        deadline = (time.monotonic() + timeout_s) if timeout_s > 0 else None
        last_note, note_every_s = 0.0, 5.0
        while deadline is None or time.monotonic() < deadline:
            frame = snapshot_of()
            engaged = frame is not None and frame.state == wire.SysState.ENGAGED
            granted = frame is not None and any(
                v == AUTHORITY_POLICY for v in (frame.authority or {}).values())
            if engaged and granted:
                return True

            now = time.monotonic()
            if now - last_note >= note_every_s:
                last_note = now
                if frame is None:
                    print("[SystemArbitrator] waiting: no SceneObjectsMsg from the avatar. "
                          "Is the sim running, and is avatar.scene_objects pointed at this "
                          "host/port?")
                elif not engaged:
                    print(f"[SystemArbitrator] waiting: avatar SysState is {frame.state}, "
                          f"need ENGAGED ({int(wire.SysState.ENGAGED)}). Press START then "
                          f"ENGAGE in the VR interface.")
                else:
                    print("[SystemArbitrator] waiting: sim ENGAGED, no arm handed over yet. "
                          "Arm the intervention panel (DAGGER), then press RESUME.")
            time.sleep(poll_s)
        print("[SystemArbitrator] timed out waiting for the operator to hand over an arm")
        return False

    def _confirm_cli(self, mode: str) -> bool:
        """Default confirmation: a blocking terminal prompt. Pass confirm= to
        the constructor for a GUI/non-interactive gate instead."""
        reply = input(f"[SystemArbitrator] About to engage in '{mode}' mode -- policy will command "
                       f"the arms directly. Type 'yes' to confirm: ")
        return reply.strip().lower() == "yes"

    def engage(self, homing_timeout_s: float = 30.0, engage_timeout_s: float = 10.0) -> bool:
        """Walks IDLE -> HOMING -> AWAITING -> ENGAGED, confirming first if
        policy_mode requires it. Returns False (and never requests ENGAGED)
        if confirmation is required and refused, or if a step times out.

        Refuses outright in intervention mode. There the operator owns SysState:
        two clients both writing cmd_requested_ is last-writer-wins, and the one
        wearing the headset is the one who can see the robot. Use claim_hold()
        plus await_operator_handover() instead.
        """
        if self.policy_mode == INTERVENTION_MODE:
            raise RuntimeError(
                "SystemArbitrator.engage() must not be called in intervention mode -- "
                "the operator owns SysState. Use claim_hold() + await_operator_handover().")

        if self.policy_mode in self._require_confirmation_for:
            if not self._confirm(self.policy_mode):
                # Said out loud so a refusal is never confused with a timeout.
                # Both used to surface as one line in run.py that named both
                # causes and distinguished neither, which sent a real transport
                # bug looking like a mistyped 'yes' for an evening.
                print("[SystemArbitrator] confirmation refused -- not engaging")
                return False

        self.client.request_state(wire.SysState.HOMING)
        if not self._wait_for(wire.SysState.AWAITING, homing_timeout_s):
            return False

        self.client.request_state(wire.SysState.ENGAGED)
        return self._wait_for(wire.SysState.ENGAGED, engage_timeout_s)

    def disengage(self, wait_s: float = 5.0) -> bool:
        """Requests IDLE; always allowed, no confirmation needed.

        Waits for the sim to actually reach IDLE by default. The sequencing
        note at the top of this module cuts both ways: because an out-of-order
        request is a silent no-op rather than an error, firing IDLE and then
        immediately HOMING (as the console's 'home' used to) means the HOMING
        request can land while the sim is still ENGAGED, get dropped, and leave
        engage() waiting on an AWAITING that will never come -- reported as
        "failed or refused" with nothing in the log to say why.

        Returns whether IDLE was observed; pass wait_s=0 to fire and forget.

        No-op in intervention mode, and this one matters on the way out rather
        than on the way in: SysState transitions end episodes on the avatar
        (markEpisodeEnd("operator_idle") on ENGAGED -> IDLE), so this running
        from run.py's finally block would close the operator's episode the
        moment the Python process exits -- including on a Ctrl+C they did not
        press. The arms are parked with claim_hold() instead, which stops the
        policy without touching the session.
        """
        if self.policy_mode == INTERVENTION_MODE:
            print("[SystemArbitrator] intervention mode: leaving SysState to the operator")
            return True

        self.client.request_state(wire.SysState.IDLE)
        if wait_s <= 0:
            return True
        return self._wait_for(wire.SysState.IDLE, wait_s)

    def _wait_for(self, target_state: int, timeout_s: float) -> bool:
        """Polls until the avatar reports target_state, or the timeout expires.

        Says what it is waiting on every few seconds, and says what it gave up
        on. The silent version of this was a 30 s stall with no output, which
        reads exactly like a hung process -- and twice it was killed as one.
        The same reasoning as await_operator_handover's progress lines; this is
        the path that was left without them.
        """
        deadline = time.monotonic() + timeout_s
        last_note, note_every_s = time.monotonic(), 5.0
        seen = self._observed_state()
        while time.monotonic() < deadline:
            seen = self._observed_state()
            if seen == target_state:
                return True
            now = time.monotonic()
            if now - last_note >= note_every_s:
                last_note = now
                if seen == wire.SysState.OFFLINE and self._state_source is not None:
                    print("[SystemArbitrator] waiting: no SceneObjectsMsg from the avatar. "
                          "Is the sim running, and is avatar.scene_objects pointed at this "
                          "host/port?")
                else:
                    print(f"[SystemArbitrator] waiting: avatar SysState is {int(seen)}, "
                          f"need {int(target_state)}")
            time.sleep(0.05)
        print(f"[SystemArbitrator] timed out after {timeout_s:.0f}s waiting for SysState "
              f"{int(target_state)} -- last seen {int(seen)}")
        return False
