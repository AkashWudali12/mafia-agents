from __future__ import annotations

from typing import Protocol

from contracts import Action, Observation


class Policy(Protocol):
    def act(self, observation: Observation) -> Action:
        """Return one structured action for the current actor."""
