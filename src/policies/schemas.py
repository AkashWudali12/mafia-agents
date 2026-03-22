from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from contracts import ActionType, DiscussionIntent


class ModelActionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    target: int | None = None
    intent: DiscussionIntent | None = None
    message: str | None = None
