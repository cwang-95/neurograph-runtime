"""ZenBrain event ledger and weak retrieval prior for Graph 3.0."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from .store import Graph3Store


class ZenBrainEventType(StrEnum):
    RETRIEVED = "retrieved"
    SELECTED = "selected"
    CITED = "cited"
    FOLLOWED_UP = "followed_up"
    USER_CONFIRMED = "user_confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"


_EVENT_WEIGHTS: dict[ZenBrainEventType, float] = {
    ZenBrainEventType.RETRIEVED: 0.0,
    ZenBrainEventType.SELECTED: 0.10,
    ZenBrainEventType.CITED: 0.18,
    ZenBrainEventType.FOLLOWED_UP: 0.22,
    ZenBrainEventType.USER_CONFIRMED: 0.35,
    ZenBrainEventType.CORRECTED: -0.35,
    ZenBrainEventType.REJECTED: -0.20,
}


class ZenBrainEventLedger:
    """Append-only events plus a bounded, read-only-at-retrieval prior.

    Recording a ``retrieved`` event has zero score impact. Retrieval itself
    never calls ``record_event``; the caller must explicitly report selection,
    citation, feedback, or correction.
    """

    def __init__(self, store: Graph3Store, *, horizon_days: float = 30.0):
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        self.store = store
        self.horizon_days = horizon_days

    def record_event(
        self,
        observation_id: str,
        event_type: ZenBrainEventType,
        *,
        query: str | None = None,
        caller: str | None = None,
        path_id: str | None = None,
        created_at: datetime | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        return self.store.record_zenbrain_event(
            target_type="observation",
            target_id=observation_id,
            observation_id=observation_id,
            event_type=event_type.value,
            query=query,
            caller=caller,
            path_id=path_id,
            created_at=created_at,
            payload=payload,
        )

    def record_pack_event(
        self,
        pack: Any,
        event_type: ZenBrainEventType,
        *,
        caller: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        count = 0
        for item in pack.evidence:
            self.record_event(
                item.observation_id,
                event_type,
                query=pack.query,
                caller=caller,
                payload=payload,
            )
            count += 1
        return count

    def score(
        self,
        observation_ids: list[str],
        *,
        now: datetime | None = None,
    ) -> dict[str, float]:
        current = now or datetime.now(timezone.utc)
        scores = {observation_id: 0.0 for observation_id in observation_ids}
        for event in self.store.zenbrain_event_history(observation_ids):
            try:
                event_type = ZenBrainEventType(event["event_type"])
                occurred_at = datetime.fromisoformat(event["created_at"])
            except (ValueError, TypeError):
                continue
            if occurred_at.tzinfo is None:
                occurred_at = occurred_at.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (current - occurred_at).total_seconds() / 86400.0)
            decay = math.exp(-age_days / self.horizon_days)
            scores[event["target_id"]] = scores.get(event["target_id"], 0.0) + _EVENT_WEIGHTS[event_type] * decay
        return {observation_id: max(-1.0, min(1.0, score)) for observation_id, score in scores.items()}
