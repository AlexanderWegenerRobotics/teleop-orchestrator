"""Operator state as the VR interface reports it (operator_state envelope, clutch per hand).
Received by GazeReceiver, which owns the local peer port the interface sends both messages to."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OperatorState:
    clutch_left: bool
    clutch_right: bool
    engaged: bool
    timestamp_ns: int
    arrival_ns: int
