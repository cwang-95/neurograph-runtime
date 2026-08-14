"""ZenBrain event ledger and weak retrieval prior for Graph 3.0."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Sequence

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

    def __init__(self, store: Graph3Store, *, horizon_days: float = 30.0, fsrs: Any | None = None):
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        self.store = store
        self.horizon_days = horizon_days
        self.fsrs = fsrs

    _FSRS_QUALITY: dict[ZenBrainEventType, int] = {
        ZenBrainEventType.SELECTED: 3,
        ZenBrainEventType.CITED: 4,
        ZenBrainEventType.FOLLOWED_UP: 4,
        ZenBrainEventType.USER_CONFIRMED: 4,
    }

    def _update_fsrs(self, observation_id: str, event_type: ZenBrainEventType) -> None:
        if self.fsrs is None or event_type not in self._FSRS_QUALITY:
            return
        memory = self.store.get_zenbrain_scheduler("observation", observation_id)
        if memory is None:
            memory = self.fsrs.new_memory()
        updated = self.fsrs.recall(memory, self._FSRS_QUALITY[event_type])
        self.store.put_zenbrain_scheduler(
            target_type="observation",
            target_id=observation_id,
            scheduler=updated,
        )

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
        event_id = self.store.record_zenbrain_event(
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
        self._update_fsrs(observation_id, event_type)
        return event_id

    def record_pack_event(
        self,
        pack: Any,
        event_type: ZenBrainEventType,
        *,
        caller: str | None = None,
        payload: dict[str, Any] | None = None,
        observation_ids: Sequence[str] | None = None,
    ) -> int:
        count = 0
        for item in pack.evidence:
            if observation_ids is not None and item.observation_id not in observation_ids:
                continue
            self.record_event(
                item.observation_id,
                event_type,
                query=pack.query,
                caller=caller,
                payload=payload,
            )
            count += 1
        return count

    def record_feedback(
        self,
        observation_ids: Sequence[str],
        event_type: ZenBrainEventType,
        *,
        query: str | None = None,
        caller: str | None = None,
        path_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Explicit answer-layer feedback; retrieval never calls this."""
        count = 0
        for observation_id in observation_ids:
            self.record_event(
                observation_id,
                event_type,
                query=query,
                caller=caller,
                path_id=path_id,
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
        if self.fsrs is not None:
            for observation_id in observation_ids:
                memory = self.store.get_zenbrain_scheduler("observation", observation_id)
                if memory is None:
                    continue
                try:
                    retrievability = self.fsrs.decay(memory)
                except Exception:
                    continue
                scores[observation_id] += 0.05 * (retrievability - 0.5)
        return {observation_id: max(-1.0, min(1.0, score)) for observation_id, score in scores.items()}
