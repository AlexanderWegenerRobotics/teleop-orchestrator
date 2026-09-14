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
from .live.sys_state_client import SysStateClient


class SystemArbitrator:
    """Sequences SysState transitions via SysStateClient, gated by the
    configured policy mode. Requires an explicit confirmation callback before
    entering any mode in require_confirmation_for -- by default just
    "autonomous", so ENGAGED is never entered in that mode by accident."""

    def __init__(self, client: SysStateClient, policy_mode: str,
                 require_confirmation_for=("autonomous",),
                 confirm: Optional[Callable[[str], bool]] = None):
        self.client = client
        self.policy_mode = policy_mode
        self._require_confirmation_for = set(require_confirmation_for)
        self._confirm = confirm or self._confirm_cli

    @property
    def autonomous_allowed(self) -> bool:
        """Whether policy is configured to command the arms directly this session."""
        return self.policy_mode == "autonomous"

    def _confirm_cli(self, mode: str) -> bool:
        """Default confirmation: a blocking terminal prompt. Pass confirm= to
        the constructor for a GUI/non-interactive gate instead."""
        reply = input(f"[SystemArbitrator] About to engage in '{mode}' mode -- policy will command "
                       f"the arms directly. Type 'yes' to confirm: ")
        return reply.strip().lower() == "yes"

    def engage(self, homing_timeout_s: float = 30.0, engage_timeout_s: float = 10.0) -> bool:
        """Walks IDLE -> HOMING -> AWAITING -> ENGAGED, confirming first if
        policy_mode requires it. Returns False (and never requests ENGAGED)
        if confirmation is required and refused, or if a step times out."""
        if self.policy_mode in self._require_confirmation_for:
            if not self._confirm(self.policy_mode):
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
        """
        self.client.request_state(wire.SysState.IDLE)
        if wait_s <= 0:
            return True
        return self._wait_for(wire.SysState.IDLE, wait_s)

    def _wait_for(self, target_state: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.client.state == target_state:
                return True
            time.sleep(0.05)
        return False
