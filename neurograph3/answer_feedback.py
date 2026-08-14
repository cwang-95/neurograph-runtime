"""Answer-layer feedback boundary for Codex and OpenClaw callers."""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import stable_id
from .retrieval import EvidencePack
from .store import Graph3Store
from .zenbrain import ZenBrainEventLedger, ZenBrainEventType


class AnswerEvidenceContext(BaseModel):
    """The addressable IDs from one EvidencePack, without copying its text."""

    model_config = ConfigDict(extra="forbid")

    query: str
    observation_ids: list[str] = Field(default_factory=list)
    claim_version_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    path_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_pack(cls, pack: EvidencePack) -> "AnswerEvidenceContext":
        return cls(
            query=pack.query,
            observation_ids=[
                item.observation_id
                for item in [*pack.evidence, *pack.context_evidence]
            ],
            claim_version_ids=[
                claim_id
                for item in pack.evidence
                for claim_id in item.claim_version_ids
            ],
            relation_ids=[
                edge["relation_id"]
                for path in pack.graph_paths
                for edge in path.get("path_edges", [])
                if edge.get("relation_id")
            ],
            path_ids=[path["path_id"] for path in pack.graph_paths if path.get("path_id")],
        )

    @model_validator(mode="after")
    def deduplicate_ids(self) -> "AnswerEvidenceContext":
        for field_name in ("observation_ids", "claim_version_ids", "relation_ids", "path_ids"):
            values = getattr(self, field_name)
            setattr(self, field_name, list(dict.fromkeys(values)))
        return self


class FeedbackRequest(BaseModel):
    """Explicit feedback submitted after an answer has used an EvidencePack."""

    model_config = ConfigDict(extra="forbid")

    context: AnswerEvidenceContext
    event_type: ZenBrainEventType
    observation_ids: list[str] = Field(default_factory=list)
    claim_version_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    path_ids: list[str] = Field(default_factory=list)
    caller: str = "answer-layer"
    query: str | None = None
    feedback_id: str | None = None
    propagate_to_observations: bool = False

    @model_validator(mode="after")
    def validate_targets(self) -> "FeedbackRequest":
        targets = self.observation_ids + self.claim_version_ids + self.relation_ids + self.path_ids
        if not targets:
            raise ValueError("feedback must select at least one target")
        allowed = {
            "observation_ids": set(self.context.observation_ids),
            "claim_version_ids": set(self.context.claim_version_ids),
            "relation_ids": set(self.context.relation_ids),
            "path_ids": set(self.context.path_ids),
        }
        for field_name, permitted in allowed.items():
            invalid = sorted(set(getattr(self, field_name)) - permitted)
            if invalid:
                raise ValueError(f"{field_name} contains IDs outside the EvidencePack: {invalid}")
        if self.query is None:
            self.query = self.context.query
        return self

    @classmethod
    def from_pack(
        cls,
        pack: EvidencePack,
        event_type: ZenBrainEventType,
        *,
        observation_ids: Sequence[str] = (),
        claim_version_ids: Sequence[str] = (),
        relation_ids: Sequence[str] = (),
        path_ids: Sequence[str] = (),
        caller: str = "answer-layer",
        feedback_id: str | None = None,
        propagate_to_observations: bool = False,
    ) -> "FeedbackRequest":
        return cls(
            context=AnswerEvidenceContext.from_pack(pack),
            event_type=event_type,
            observation_ids=list(observation_ids),
            claim_version_ids=list(claim_version_ids),
            relation_ids=list(relation_ids),
            path_ids=list(path_ids),
            caller=caller,
            feedback_id=feedback_id,
            propagate_to_observations=propagate_to_observations,
        )


class FeedbackResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    event_type: ZenBrainEventType
    recorded_events: int
    observation_events: int
    claim_events: int
    relation_events: int
    path_events: int


class AnswerFeedbackRecorder:
    """Validate and persist one answer-layer feedback request."""

    def __init__(self, store: Graph3Store, ledger: ZenBrainEventLedger | None = None):
        self.store = store
        self.ledger = ledger or ZenBrainEventLedger(store)

    @staticmethod
    def _feedback_id(request: FeedbackRequest) -> str:
        return request.feedback_id or stable_id(
            "answer_feedback",
            {
                "query": request.query,
                "event_type": request.event_type.value,
                "observation_ids": request.observation_ids,
                "claim_version_ids": request.claim_version_ids,
                "relation_ids": request.relation_ids,
                "path_ids": request.path_ids,
            },
        )

    def record(self, request: FeedbackRequest) -> FeedbackResult:
        feedback_id = self._feedback_id(request)
        before_feedback_counts = self.store.zenbrain_feedback_counts(feedback_id)
        payload: dict[str, Any] = {
            "feedback_id": feedback_id,
            "source": "answer-layer",
        }
        propagated_observation_ids: set[str] = set()
        if request.propagate_to_observations and request.claim_version_ids:
            linked = self.store.observation_ids_for_claim_versions(request.claim_version_ids)
            allowed = set(request.context.observation_ids)
            propagated_observation_ids = {
                observation_id
                for observation_ids in linked.values()
                for observation_id in observation_ids
                if observation_id in allowed
            }

        direct_observation_ids = [
            observation_id
            for observation_id in request.observation_ids
            if observation_id not in propagated_observation_ids
        ]
        self.ledger.record_feedback(
            direct_observation_ids,
            request.event_type,
            query=request.query,
            caller=request.caller,
            payload=payload,
        )
        self.ledger.record_claim_feedback(
            request.claim_version_ids,
            request.event_type,
            query=request.query,
            caller=request.caller,
            payload=payload,
            propagate_to_observations=request.propagate_to_observations,
            allowed_observation_ids=request.context.observation_ids,
        )
        self.ledger.record_feedback(
            [],
            request.event_type,
            query=request.query,
            caller=request.caller,
            payload=payload,
            relation_ids=request.relation_ids,
            path_ids=request.path_ids,
        )
        after_feedback_counts = self.store.zenbrain_feedback_counts(feedback_id)
        new_counts = {
            target_type: max(
                0,
                after_feedback_counts.get(target_type, 0) - before_feedback_counts.get(target_type, 0),
            )
            for target_type in set(before_feedback_counts) | set(after_feedback_counts)
        }
        return FeedbackResult(
            feedback_id=feedback_id,
            event_type=request.event_type,
            recorded_events=sum(new_counts.values()),
            observation_events=new_counts.get("observation", 0),
            claim_events=new_counts.get("claim", 0),
            relation_events=new_counts.get("relation", 0),
            path_events=new_counts.get("path", 0),
        )


__all__ = [
    "AnswerEvidenceContext",
    "AnswerFeedbackRecorder",
    "FeedbackRequest",
    "FeedbackResult",
]
