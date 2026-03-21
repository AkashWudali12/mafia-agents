from __future__ import annotations

from typing import Protocol

from contracts import Action, ActionType, DiscussionIntent, Observation


class Policy(Protocol):
    def act(self, observation: Observation) -> Action:
        """Return one structured action for the current actor."""


class FirstLegalPolicy:
    def act(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            return Action(actor=observation.actor, action_type=ActionType.NOOP)
        choice = observation.legal_actions[0]
        target = choice.legal_targets[0] if choice.legal_targets else None
        intent = choice.legal_intents[0] if choice.legal_intents else None
        message = "holding a position" if choice.action_type == ActionType.SPEAK else None
        if choice.action_type == ActionType.SPEAK and intent is None:
            intent = DiscussionIntent.QUESTION
        return Action(
            actor=observation.actor,
            action_type=choice.action_type,
            target=target,
            intent=intent,
            message=message,
        )
