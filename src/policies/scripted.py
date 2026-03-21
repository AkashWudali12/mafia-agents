from __future__ import annotations

from contracts import Action, ActionType, DiscussionIntent, LegalActionSpec, Observation

from .interfaces import Policy


class ScriptedVillagerPolicy(Policy):
    def act(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            return Action(actor=observation.actor, action_type=ActionType.NOOP)

        choice = observation.legal_actions[0]
        if choice.action_type == ActionType.SPEAK:
            target = self._preferred_target(observation, choice)
            intent = (
                DiscussionIntent.ACCUSE
                if DiscussionIntent.ACCUSE in choice.legal_intents
                else choice.legal_intents[0]
                if choice.legal_intents
                else None
            )
            return Action(
                actor=observation.actor,
                action_type=ActionType.SPEAK,
                target=target,
                intent=intent,
                message="sharing suspicion",
            )
        if choice.action_type == ActionType.VOTE:
            return Action(
                actor=observation.actor,
                action_type=ActionType.VOTE,
                target=self._preferred_target(observation, choice),
            )
        return Action(actor=observation.actor, action_type=choice.action_type)

    def _preferred_target(self, observation: Observation, choice: LegalActionSpec) -> int | None:
        others = tuple(target for target in choice.legal_targets if target != observation.actor)
        if others:
            return min(others)
        if choice.legal_targets:
            return choice.legal_targets[0]
        return None


class ScriptedDoctorPolicy(Policy):
    def __init__(self) -> None:
        self._day_policy = ScriptedVillagerPolicy()

    def act(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            return Action(actor=observation.actor, action_type=ActionType.NOOP)

        choice = observation.legal_actions[0]
        if choice.action_type == ActionType.PROTECT:
            target = (
                observation.actor
                if observation.actor in choice.legal_targets
                else self._preferred_target(observation, choice)
            )
            return Action(
                actor=observation.actor,
                action_type=ActionType.PROTECT,
                target=target,
            )
        return self._day_policy.act(observation)

    def _preferred_target(self, observation: Observation, choice: LegalActionSpec) -> int | None:
        others = tuple(target for target in choice.legal_targets if target != observation.actor)
        if others:
            return min(others)
        if choice.legal_targets:
            return choice.legal_targets[0]
        return None


class ScriptedDetectivePolicy(Policy):
    def __init__(self) -> None:
        self._day_policy = ScriptedVillagerPolicy()

    def act(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            return Action(actor=observation.actor, action_type=ActionType.NOOP)

        choice = observation.legal_actions[0]
        if choice.action_type == ActionType.INVESTIGATE:
            investigated = {result.target for result in observation.private_state.investigation_results}
            unseen_targets = tuple(target for target in choice.legal_targets if target not in investigated)
            target_pool = unseen_targets or choice.legal_targets
            return Action(
                actor=observation.actor,
                action_type=ActionType.INVESTIGATE,
                target=min(target_pool) if target_pool else None,
            )
        return self._day_policy.act(observation)


class ScriptedMafiaPolicy(Policy):
    def __init__(self) -> None:
        self._day_policy = ScriptedVillagerPolicy()

    def act(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            return Action(actor=observation.actor, action_type=ActionType.NOOP)

        choice = observation.legal_actions[0]
        if choice.action_type == ActionType.NIGHT_KILL:
            targets = tuple(target for target in choice.legal_targets if target != observation.actor)
            target_pool = targets or choice.legal_targets
            return Action(
                actor=observation.actor,
                action_type=ActionType.NIGHT_KILL,
                target=min(target_pool) if target_pool else None,
            )
        return self._day_policy.act(observation)
