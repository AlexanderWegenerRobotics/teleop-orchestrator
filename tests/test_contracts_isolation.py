"""Guards that contracts stays a leaf: modules can depend on it without pulling in the orchestrator."""

import sys


def test_contracts_has_no_parent_imports():
    """Importing contracts must not load any non-contracts orchestrator module."""
    for m in list(sys.modules):
        if m.startswith("teleop_orchestrator"):
            del sys.modules[m]

    import teleop_orchestrator.contracts  # noqa: F401

    leaked = [
        m for m in sys.modules
        if m.startswith("teleop_orchestrator")
        and not m.startswith("teleop_orchestrator.contracts")
        and m != "teleop_orchestrator"
    ]
    assert not leaked, f"contracts leaked parent imports: {leaked}"


def test_public_surface_imports():
    """The advertised public types are importable from the package root."""
    from teleop_orchestrator.contracts import (  # noqa: F401
        SensorFrame, Module, SensorModule,
        IntentOutput, ActionOutput, AssistOutput, ArmIntent, Phase, NULL_TARGET,
    )