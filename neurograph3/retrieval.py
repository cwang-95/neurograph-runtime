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
    vector_score: float | None = None
    graph_score: float | None = None
    combined_score: float = 0.0
    retrieval_routes: list[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    query_plan: QueryPlan
    slot_status: dict[str, SlotStatus]
    evidence: list[EvidenceItem]
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)


class Graph3Retriever:
    def __init__(self, store: Graph3Store, embedder: Any | None = None):
        self.store = store
        self.embedder = embedder

    def index_vectors(self) -> int:
        if self.embedder is None:
            return 0
        observations = self.store.list_observations()
        batch_size = max(int(getattr(self.embedder, "batch_size", 16)), 1)
        indexed = 0
        for start in range(0, len(observations), batch_size):
            batch = observations[start : start + batch_size]
            texts = [item["value"] for item in batch]
            try:
                vectors = self.embedder.embed(texts)
            except Exception:
                if len(texts) == 1:
                    raise
                vectors = []
                for text in texts:
                    vectors.extend(self.embedder.embed([text]))
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embedding count mismatch: requested {len(batch)}, got {len(vectors)}"
                )
            self.store.put_embeddings(
                getattr(self.embedder, "model", "unknown"),
                {item["observation_id"]: vector for item, vector in zip(batch, vectors)},
            )
            indexed += len(vectors)
        return indexed

    def retrieve(self, query: str, limit: int = 8) -> EvidencePack:
        plan = QueryPlan.from_query(query)
        candidates: dict[str, dict[str, Any]] = {}
        trace: dict[str, Any] = {
            "routes": ["lexical", "numeric", "entity", "graph"],
            "completion_model_called": False,
        }

        def add_hit(hit: dict[str, Any], route: str, score: float | None = None) -> None:
            item = candidates.setdefault(hit["observation_id"], dict(hit))
            routes = set(item.get("retrieval_routes", []))
            routes.add(route)
            item["retrieval_routes"] = sorted(routes)
            if route == "vector":
                item["vector_score"] = max(float(item.get("vector_score") or 0.0), float(score or 0.0))
            elif route == "graph":
                item["graph_score"] = max(float(item.get("graph_score") or 0.0), float(score or 0.0))

        lexical_hits = self.store.search_lexical(query, limit=min(limit * 3, plan.max_candidates))
        for hit in lexical_hits:
            add_hit(hit, "lexical")

        entity_hits = self.store.search_entities(query, limit=10)
        entity_ids = [item["entity_id"] for item in entity_hits]
        graph_paths, graph_hits = self.store.expand_graph(entity_ids, max_hops=1)
        for hit in graph_hits:
            path_score = max(
                (path["confidence"] for path in graph_paths if hit["observation_id"] in path["observation_ids"]),
                default=0.0,
            )
            add_hit(hit, "graph", path_score)

        if self.embedder is not None:
            try:
                query_vector = self.embedder.embed([query])[0]
                vector_hits = self.store.search_vector(query_vector, limit=min(limit * 3, plan.max_candidates))
                for hit in vector_hits:
                    add_hit(hit, "vector", hit.get("vector_score"))
                trace["routes"].append("vector")
                trace["vector_candidates"] = len(vector_hits)
            except Exception as exc:  # keep lexical/graph retrieval available
                trace["vector_error"] = str(exc)

        for item in candidates.values():
            item["combined_score"] = (
                float(item.get("lexical_score", 0.0))
                + float(item.get("vector_score", 0.0) or 0.0)
                + float(item.get("graph_score", 0.0) or 0.0) * 0.5
            )
        hits = sorted(candidates.values(), key=lambda item: item["combined_score"], reverse=True)[:limit]
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
        plan = plan.model_copy(update={"routes": trace["routes"]})
        return EvidencePack(
            query=query,
            query_plan=plan,
            slot_status=slot_status,
            evidence=evidence,
            graph_paths=graph_paths,
            missing=missing,
            citations=citations,
            retrieval_trace={**trace, "candidate_count": len(candidates), "entity_seed_count": len(entity_ids)},
        )
