"""Phase-2 query planning and EvidencePack baseline."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .store import Graph3Store


class SlotStatus(StrEnum):
    SUPPORTED = "supported"
    CONFLICTED = "conflicted"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    LOW_CONFIDENCE = "low_confidence"


class EvidenceSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    required: bool = True
    status: SlotStatus = SlotStatus.MISSING


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    query_types: list[str]
    numeric_constraints: list[str] = Field(default_factory=list)
    evidence_slots: list[EvidenceSlot]
    routes: list[str]
    max_candidates: int = Field(default=20, ge=1)

    @classmethod
    def from_query(cls, query: str) -> "QueryPlan":
        lowered = query.casefold()
        query_types = ["exact_fact"] if re.search(r"\d", query) else ["exploratory"]
        if any(word in lowered for word in ("方法", "流程", "机制", "怎么")):
            query_types.append("mechanism")
        if any(word in lowered for word in ("结果", "效果", "指标", "准确", "性能")):
            query_types.append("quantitative_result")
        numeric_constraints = re.findall(r"\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?", query)
        slots = [EvidenceSlot(name="direct_evidence")]
        if "mechanism" in query_types:
            slots.append(EvidenceSlot(name="mechanism"))
        if "quantitative_result" in query_types:
            slots.append(EvidenceSlot(name="quantitative_result"))
        return cls(
            query=query,
            query_types=query_types,
            numeric_constraints=numeric_constraints,
            evidence_slots=slots,
            routes=["lexical", "numeric"],
        )


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    element_id: str
    aligned_element_ids: list[str]
    kind: str
    value: str
    asset_id: str
    element_type: str
    locator: dict[str, Any]
    lexical_score: float
    matched_terms: int
    matched_numbers: int


class EvidencePack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    query_plan: QueryPlan
    slot_status: dict[str, SlotStatus]
    evidence: list[EvidenceItem]
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)


class Graph3Retriever:
    def __init__(self, store: Graph3Store):
        self.store = store

    def retrieve(self, query: str, limit: int = 8) -> EvidencePack:
        plan = QueryPlan.from_query(query)
        hits = self.store.search_lexical(query, limit=min(limit, plan.max_candidates))
        evidence = [EvidenceItem.model_validate(hit) for hit in hits]
        direct_status = SlotStatus.SUPPORTED if evidence else SlotStatus.MISSING
        slot_status = {slot.name: direct_status for slot in plan.evidence_slots}
        missing = [name for name, status in slot_status.items() if status is SlotStatus.MISSING]
        citations = [
            {
                "observation_id": item.observation_id,
                "asset_id": item.asset_id,
                "element_id": item.element_id,
                "aligned_element_ids": item.aligned_element_ids,
                "locator": item.locator,
            }
            for item in evidence
        ]
        return EvidencePack(
            query=query,
            query_plan=plan,
            slot_status=slot_status,
            evidence=evidence,
            missing=missing,
            citations=citations,
            retrieval_trace={
                "routes": ["lexical", "numeric"],
                "candidate_count": len(hits),
                "completion_model_called": False,
            },
        )
