"""Shared lifecycle contract for all runtime modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .frame import SensorFrame

In = TypeVar("In")
Out = TypeVar("Out")


class Module(ABC, Generic[In, Out]):
    """Base for any runtime module; online and causal, state kept between steps."""

    @abstractmethod
    def reset(self) -> None:
        """Clears internal state at an episode boundary; called before the first step()."""
        ...

    @abstractmethod
    def step(self, x: In) -> Out:
        """Consumes one tick of input and returns this module's output, using only info up to now."""
        ...

    def load(self, path: str) -> None:
        """Loads any parameters/checkpoints; no-op for modules without state to restore."""
        pass

    @property
    def name(self) -> str:
        """Short identifier used in logs and results."""
        return type(self).__name__


class SensorModule(Module[SensorFrame, Out], Generic[Out]):
    """A producer module that reads the raw SensorFrame directly (intent, IL)."""
    ...