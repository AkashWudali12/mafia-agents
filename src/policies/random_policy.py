from __future__ import annotations

import random

from contracts import Action, ActionType, DiscussionIntent, LegalActionSpec, Observation

from .interfaces import Policy


class RandomLegalPolicy(Policy):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            return Action(actor=observation.actor, action_type=ActionType.NOOP)

        choice = self._rng.choice(observation.legal_actions)
        target = self._choose_target(choice)
        intent = self._choose_intent(choice)
        message = "random-policy" if choice.action_type == ActionType.SPEAK and choice.allow_message else None
        return Action(
            actor=observation.actor,
            action_type=choice.action_type,
            target=target,
            intent=intent,
            message=message,
        )

    def _choose_target(self, choice: LegalActionSpec) -> int | None:
        if not choice.legal_targets:
            return None
        return self._rng.choice(choice.legal_targets)

    def _choose_intent(self, choice: LegalActionSpec) -> DiscussionIntent | None:
        if not choice.legal_intents:
            return None
        return self._rng.choice(choice.legal_intents)
