"""The reusable inference loop: source -> modules -> run log. Same path offline, sim, hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .contracts import SensorFrame, Module
from .logging import RunLogger


@dataclass
class Wire:
    """Declares that a registered module is a consumer, not a producer: its
    step() input is assembled from the raw frame plus named upstream modules'
    outputs, rather than the frame alone. depends_on names must already be
    registered (and therefore already stepped this tick -- see run()'s
    registration-order requirement)."""

    depends_on: list[str]
    assemble: Callable[[SensorFrame, dict[str, Any]], Any]


class Orchestrator:
    """Runs registered modules over a stream of SensorFrames, one tick at a time.

    The source is anything iterable of SensorFrame (ReplaySource offline; LiveSource
    over the sim/hardware transports). Modules run in registration order; their
    outputs are collected per tick. By default every module's step() receives the
    raw frame (Module[SensorFrame, Out], i.e. SensorModule) -- pass a Wire in
    `wiring` for a consumer whose input should instead be assembled from other
    modules' outputs (e.g. a recommender reading the intent module's IntentOutput).
    A wired consumer must be registered after everything it depends_on.
    """

    def __init__(self, source: Iterable[SensorFrame], modules: dict[str, Module],
                 wiring: dict[str, Wire] | None = None):
        self.source = source
        self.modules = modules
        self.wiring = wiring or {}
        for name, wire in self.wiring.items():
            registered = list(modules)
            missing = [d for d in wire.depends_on if d not in registered]
            if missing:
                raise ValueError(f"{name}'s wiring depends on unregistered module(s): {missing}")
            if registered.index(name) < max(registered.index(d) for d in wire.depends_on):
                raise ValueError(f"{name} must be registered after its dependencies {wire.depends_on}")

    def run(self, logger: RunLogger | None = None,
            on_tick: Callable[[SensorFrame, dict[str, Any]], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> RunLogger:
        """Resets modules, steps every frame through each module, and records the run.

        on_tick, if given, is called with (frame, outputs) after every tick's
        outputs are computed but before the next frame is pulled -- the hook
        for anything that needs to act on an output rather than just log it
        (e.g. sending policy's ActionOutput to the arms, gated by whatever
        mode/arbitration decides that's allowed this session). Deliberately
        not a Module: actuation is a side effect on the world, not an
        inference step with its own output to compose further.

        should_stop, if given, is polled once per tick (before stepping any
        module this tick) -- the graceful-stop hook for a live session (e.g.
        an operator console's "stop" command). Note the loop can only check
        it between frames: if the source is blocked waiting for the next one
        (LiveSource polling for a tick that hasn't arrived), stop takes
        effect on the next frame it does produce, not instantly.
        """
        logger = logger or RunLogger()
        for m in self.modules.values():
            m.reset()
        for frame in self.source:
            if should_stop is not None and should_stop():
                break
            outputs: dict[str, Any] = {}
            for name, m in self.modules.items():
                wire = self.wiring.get(name)
                if wire is None:
                    x = frame
                else:
                    upstream = {dep: outputs[dep] for dep in wire.depends_on}
                    x = wire.assemble(frame, upstream)
                outputs[name] = m.step(x)
            logger.record(frame, outputs)
            if on_tick is not None:
                on_tick(frame, outputs)
        return logger
