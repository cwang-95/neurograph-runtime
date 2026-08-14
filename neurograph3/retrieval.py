"""Phase-2 query planning and EvidencePack baseline."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .store import Graph3Store


_MECHANISM_MARKERS = (
    "机制", "流程", "framework", "architecture", "component", "module",
    "input", "output", "step", "network", "首先", "然后", "最后",
)
_COMPARISON_MARKERS = ("比较", "对比", "versus", "compared", "difference", "优于", "相比")
_LIMITATION_MARKERS = ("限制", "局限", "不足", "limitation", "适用范围", "不适用")


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


class FollowUpQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    reason: str
    missing_slots: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    query_types: list[str]
    numeric_constraints: list[str] = Field(default_factory=list)
    evidence_slots: list[EvidenceSlot]
    routes: list[str]
    max_candidates: int = Field(default=20, ge=1)
    max_hops: int = Field(default=1, ge=1, le=3)
    beam_width: int = Field(default=20, ge=1, le=100)

    @classmethod
    def from_query(cls, query: str) -> "QueryPlan":
        lowered = query.casefold()
        query_types = ["exact_fact", "quantitative_result"] if re.search(r"\d", query) else ["exploratory"]
        if any(word in lowered for word in ("方法", "流程", "机制", "怎么")):
            query_types.append("mechanism")
        if any(word in lowered for word in ("结果", "效果", "指标", "准确", "性能")):
            query_types.append("quantitative_result")
        if any(word in lowered for word in ("比较", "对比", "区别", "优劣")):
            query_types.append("comparison")
        if any(word in lowered for word in ("限制", "局限", "不足", "适用范围")):
            query_types.append("limitations")
        numeric_constraints = re.findall(r"\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?", query)
        slots = [EvidenceSlot(name="direct_evidence")]
        if "mechanism" in query_types:
            slots.append(EvidenceSlot(name="mechanism"))
        if "quantitative_result" in query_types:
            slots.append(EvidenceSlot(name="quantitative_result"))
        if "comparison" in query_types:
            slots.append(EvidenceSlot(name="comparison"))
        if "limitations" in query_types:
            slots.append(EvidenceSlot(name="limitations"))
        return cls(
            query=query,
            query_types=query_types,
            numeric_constraints=numeric_constraints,
            evidence_slots=slots,
            routes=["lexical", "numeric"],
            max_hops=1 if query_types == ["exact_fact"] else 2,
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
    duplicate_group_id: str | None = None
    source_quality: float = 0.5
    lexical_score: float
    matched_terms: int
    matched_numbers: int
    vector_score: float | None = None
    graph_score: float | None = None
    zenbrain_prior: float = 0.0
    zenbrain_edge_prior: float = 0.0
    zenbrain_path_prior: float = 0.0
    claim_version_ids: list[str] = Field(default_factory=list)
    suppressed_claim_version_ids: list[str] = Field(default_factory=list)
    claim_conflict_ids: list[str] = Field(default_factory=list)
    zenbrain_claim_prior: float = 0.0
    fusion_score: float = 0.0
    route_contributions: dict[str, float] = Field(default_factory=dict)
    combined_score: float = 0.0
    retrieval_routes: list[str] = Field(default_factory=list)
    supporting_observation_ids: list[str] = Field(default_factory=list)
    supporting_citations: list[dict[str, Any]] = Field(default_factory=list)
    context_of_observation_ids: list[str] = Field(default_factory=list)
    context_relation: str | None = None
    context_distance: int | None = None


class EvidencePack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    query_plan: QueryPlan
    slot_status: dict[str, SlotStatus]
    slot_evidence: dict[str, list[str]] = Field(default_factory=dict)
    evidence: list[EvidenceItem]
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    context_evidence: list[EvidenceItem] = Field(default_factory=list)
    follow_up_required: bool = False
    follow_up_questions: list[FollowUpQuestion] = Field(default_factory=list)
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)


class Graph3Retriever:
    def __init__(
        self,
        store: Graph3Store,
        embedder: Any | None = None,
        zenbrain_prior: Any | None = None,
        vector_index: Any | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.zenbrain_prior = zenbrain_prior
        self.vector_index = vector_index

    def index_vectors(self, *, only_missing: bool = False, force: bool = False) -> int:
        if only_missing and force:
            raise ValueError("only_missing and force cannot both be enabled")
        if self.embedder is None:
            return 0
        observations = self.store.list_observations()
        model = getattr(self.embedder, "model", "unknown")
        if only_missing:
            existing_records = self.store.list_embedding_records(model=model)
            existing_ids = {record["observation_id"] for record in existing_records}
            all_models = {record["model"] for record in self.store.list_embedding_records()}
            if all_models and all_models != {model}:
                raise ValueError(
                    f"embedding model {model!r} differs from existing models {sorted(all_models)!r}; "
                    "use force=True to replace the current vector space"
                )
            observations = [
                item for item in observations if item["observation_id"] not in existing_ids
            ]
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
                model,
                {item["observation_id"]: vector for item, vector in zip(batch, vectors)},
            )
            indexed += len(vectors)
        if self.vector_index is not None and self.store.list_embedding_records():
            self.vector_index.rebuild(self.store.list_embedding_records())
        return indexed

    @staticmethod
    def _has_numeric_evidence(value: str, constraints: list[str]) -> bool:
        normalized = " ".join(value.casefold().split())
        if constraints:
            return any(" ".join(item.casefold().split()) in normalized for item in constraints)
        return bool(re.search(r"\d+(?:\.\d+)?\s*(?:ms|s|gy|cc|%|percent)", normalized))

    @staticmethod
    def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
        lowered = value.casefold()
        return any(marker.casefold() in lowered for marker in markers)

    @classmethod
    def _matches_slot(cls, plan: QueryPlan, slot: EvidenceSlot, item: EvidenceItem) -> bool:
        if slot.name == "direct_evidence":
            return True
        if slot.name == "mechanism":
            return cls._contains_any(item.value, _MECHANISM_MARKERS)
        if slot.name == "quantitative_result":
            return item.matched_numbers > 0 or cls._has_numeric_evidence(item.value, plan.numeric_constraints)
        if slot.name == "comparison":
            return cls._contains_any(item.value, _COMPARISON_MARKERS)
        if slot.name == "limitations":
            return cls._contains_any(item.value, _LIMITATION_MARKERS)
        return False

    def _evaluate_slots(
        self,
        plan: QueryPlan,
        evidence: list[EvidenceItem],
    ) -> tuple[dict[str, SlotStatus], dict[str, list[str]]]:
        statuses: dict[str, SlotStatus] = {}
        slot_evidence: dict[str, list[str]] = {}
        for slot in plan.evidence_slots:
            matched = [item for item in evidence if self._matches_slot(plan, slot, item)]
            ids = [item.observation_id for item in matched]
            slot_evidence[slot.name] = ids
            statuses[slot.name] = SlotStatus.SUPPORTED if ids else SlotStatus.MISSING
        return statuses, slot_evidence

    def _select_evidence(
        self,
        plan: QueryPlan,
        candidates: dict[str, dict[str, Any]],
        limit: int,
    ) -> list[EvidenceItem]:
        ordered = sorted(candidates.values(), key=lambda item: item["combined_score"], reverse=True)
        deduplicated: list[dict[str, Any]] = []
        representatives: dict[str, dict[str, Any]] = {}
        for candidate in ordered:
            group_id = candidate.get("duplicate_group_id") or f"observation:{candidate['observation_id']}"
            representative = representatives.get(group_id)
            if representative is None:
                representative = dict(candidate)
                representative["supporting_observation_ids"] = list(
                    representative.get("supporting_observation_ids", [])
                )
                representative["supporting_citations"] = list(
                    representative.get("supporting_citations", [])
                )
                representatives[group_id] = representative
                deduplicated.append(representative)
                continue
            representative["supporting_observation_ids"].append(candidate["observation_id"])
            representative["supporting_citations"].append(
                {
                    "observation_id": candidate["observation_id"],
                    "asset_id": candidate["asset_id"],
                    "element_id": candidate["element_id"],
                    "aligned_element_ids": candidate.get("aligned_element_ids", []),
                    "locator": candidate["locator"],
                    "source_quality": candidate.get("source_quality", 0.5),
                    "claim_version_ids": candidate.get("claim_version_ids", []),
                }
            )
            representative["claim_version_ids"] = list(
                dict.fromkeys(
                    representative.get("claim_version_ids", [])
                    + candidate.get("claim_version_ids", [])
                )
            )
        ordered = deduplicated
        selected = [EvidenceItem.model_validate(item) for item in ordered[:limit]]
        if not selected:
            return []

        for slot in plan.evidence_slots:
            if not slot.required or any(self._matches_slot(plan, slot, item) for item in selected):
                continue
            replacement = next(
                (
                    EvidenceItem.model_validate(item)
                    for item in ordered[limit:]
                    if self._matches_slot(plan, slot, EvidenceItem.model_validate(item))
                ),
                None,
            )
            if replacement is None:
                continue
            replace_index = min(range(len(selected)), key=lambda index: selected[index].combined_score)
            selected[replace_index] = replacement
        return sorted(selected, key=lambda item: item.combined_score, reverse=True)

    @staticmethod
    def _follow_up_questions(
        plan: QueryPlan,
        slot_status: dict[str, SlotStatus],
        evidence: list[EvidenceItem],
        options: list[str] | None = None,
    ) -> list[FollowUpQuestion]:
        missing = [
            slot.name
            for slot in plan.evidence_slots
            if slot.required and slot_status.get(slot.name) in {SlotStatus.MISSING, SlotStatus.LOW_CONFIDENCE, SlotStatus.CONFLICTED}
        ]
        if not missing:
            return []
        labels = {
            "direct_evidence": "直接证据",
            "mechanism": "机制或流程",
            "quantitative_result": "量化结果或指标",
            "comparison": "比较对象与差异",
            "limitations": "限制条件与适用范围",
        }
        missing_labels = [labels.get(name, name) for name in missing]
        missing_text = "、".join(missing_labels)
        if evidence:
            question = f"我找到了相关材料，但还缺少{missing_text}。你希望优先确认哪一部分？"
            reason = "当前证据与问题相关，但未覆盖所有必需证据槽位。"
        else:
            question = f"当前材料没有形成可核验的{missing_text}证据。请补充对象、范围或来源线索。"
            reason = "当前召回结果不足以支撑问题，继续生成答案会有缺项风险。"
        return [FollowUpQuestion(question=question, reason=reason, missing_slots=missing, options=options or [])]

    def _ambiguity_question(
        self,
        query: str,
        entity_ids: list[str],
        evidence: list[EvidenceItem],
    ) -> FollowUpQuestion | None:
        if entity_ids or not evidence:
            return None
        lowered = query.casefold()
        reference_markers = ("这个", "该", "它", "此方法", "这个方法", "the method", "it")
        target_markers = ("方法", "机制", "结果", "效果", "性能", "怎么", "流程")
        overview_markers = ("概览", "概述", "有哪些", "全貌", "综述", "整体")
        if not any(marker in lowered for marker in reference_markers + target_markers):
            return None

        directions: list[str] = []
        for entity in self.store.list_entities():
            names = (entity["canonical_name"], *entity["aliases"])
            if any(name.casefold() in item.value.casefold() for item in evidence for name in names):
                directions.append(entity["canonical_name"])
        directions = list(dict.fromkeys(directions))
        if len(directions) < 2 and not any(marker in lowered for marker in reference_markers):
            return None
        if any(marker in lowered for marker in overview_markers):
            return None

        if directions:
            options = directions[:5]
            question = f"当前材料涉及多个方向（{'、'.join(options)}）。你希望具体查看哪一个？"
            reason = "问题未指定实体，且候选证据来自多个可能改变答案的方向。"
        else:
            options = []
            question = "你提到的对象还不明确。请补充具体方法、模型、设备或讲座主题。"
            reason = "问题使用了指代词，但当前上下文无法唯一确定指代对象。"
        return FollowUpQuestion(question=question, reason=reason, options=options)

    def _expand_context(self, evidence: list[EvidenceItem], limit: int) -> list[EvidenceItem]:
        if not evidence or limit < 1:
            return []
        primary_ids = {item.observation_id for item in evidence}
        hits = self.store.context_hits(
            list(primary_ids),
            per_seed=2,
            limit=limit,
        )
        context_items: list[EvidenceItem] = []
        for hit in hits:
            if hit["observation_id"] in primary_ids:
                continue
            context_of = hit.pop("_context_of_observation_ids", [])
            context_relation = hit.pop("_context_relation", None)
            context_distance = hit.pop("_context_distance", None)
            context_items.append(
                EvidenceItem.model_validate(
                    {
                        **hit,
                        "context_of_observation_ids": context_of,
                        "context_relation": context_relation,
                        "context_distance": context_distance,
                        "retrieval_routes": ["context"],
                    }
                )
            )
        return context_items

    def retrieve(self, query: str, limit: int = 8) -> EvidencePack:
        plan = QueryPlan.from_query(query)
        candidates: dict[str, dict[str, Any]] = {}
        route_ranks: dict[str, dict[str, int]] = {}
        route_weights = {"lexical": 1.0, "vector": 1.0, "graph": 0.8}
        rrf_k = 60
        trace: dict[str, Any] = {
            "routes": ["lexical", "numeric", "entity", "graph"],
            "completion_model_called": False,
            "fusion": {
                "method": "rrf",
                "k": rrf_k,
                "route_weights": route_weights,
            },
        }

        def add_hit(
            hit: dict[str, Any],
            route: str,
            score: float | None = None,
            rank: int | None = None,
        ) -> None:
            item = candidates.setdefault(hit["observation_id"], dict(hit))
            routes = set(item.get("retrieval_routes", []))
            routes.add(route)
            item["retrieval_routes"] = sorted(routes)
            if rank is not None:
                route_ranks.setdefault(route, {}).setdefault(hit["observation_id"], rank)
            if route == "vector":
                item["vector_score"] = max(float(item.get("vector_score") or 0.0), float(score or 0.0))
            elif route == "graph":
                item["graph_score"] = max(float(item.get("graph_score") or 0.0), float(score or 0.0))

        lexical_hits = self.store.search_lexical(query, limit=min(limit * 3, plan.max_candidates))
        for rank, hit in enumerate(lexical_hits, start=1):
            add_hit(hit, "lexical", rank=rank)

        entity_hits = self.store.search_entities(query, limit=10)
        entity_ids = [item["entity_id"] for item in entity_hits]
        graph_paths, graph_hits = self.store.expand_graph(
            entity_ids,
            max_hops=plan.max_hops,
            beam_width=plan.beam_width,
        )
        ranked_graph_hits = sorted(
            graph_hits,
            key=lambda hit: max(
                (path["confidence"] for path in graph_paths if hit["observation_id"] in path["observation_ids"]),
                default=0.0,
            ),
            reverse=True,
        )
        for rank, hit in enumerate(ranked_graph_hits, start=1):
            path_score = max(
                (path["confidence"] for path in graph_paths if hit["observation_id"] in path["observation_ids"]),
                default=0.0,
            )
            add_hit(hit, "graph", path_score, rank=rank)

        if self.embedder is not None:
            try:
                query_vector = self.embedder.embed([query])[0]
                vector_limit = min(limit * 3, plan.max_candidates)
                if self.vector_index is None:
                    vector_hits = self.store.search_vector(query_vector, limit=vector_limit)
                else:
                    indexed_hits = self.vector_index.search(query_vector, limit=vector_limit)
                    vector_scores = {
                        item["observation_id"]: float(item["vector_score"])
                        for item in indexed_hits
                    }
                    vector_hits = self.store.observation_hits(
                        list(vector_scores), vector_scores=vector_scores
                    )
                for rank, hit in enumerate(vector_hits, start=1):
                    add_hit(hit, "vector", hit.get("vector_score"), rank=rank)
                trace["routes"].append("vector")
                trace["vector_candidates"] = len(vector_hits)
                if self.vector_index is not None:
                    trace["vector_backend"] = getattr(self.vector_index, "backend_name", "custom")
            except Exception as exc:  # keep lexical/graph retrieval available
                trace["vector_error"] = str(exc)
                if self.vector_index is not None:
                    trace["vector_fallback"] = "sqlite_bruteforce"
                    try:
                        vector_hits = self.store.search_vector(
                            query_vector, limit=min(limit * 3, plan.max_candidates)
                        )
                        for rank, hit in enumerate(vector_hits, start=1):
                            add_hit(hit, "vector", hit.get("vector_score"), rank=rank)
                        trace["routes"].append("vector")
                        trace["vector_candidates"] = len(vector_hits)
                    except Exception as fallback_exc:
                        trace["vector_fallback_error"] = str(fallback_exc)

        prior_scores: dict[str, float] = {}
        edge_prior_scores: dict[str, float] = {}
        path_prior_scores: dict[str, float] = {}
        claim_prior_scores: dict[str, float] = {}
        claim_ids_by_observation = self.store.claim_version_ids_for_observations(list(candidates))
        claim_ids = list(
            dict.fromkeys(
                claim_id
                for ids in claim_ids_by_observation.values()
                for claim_id in ids
            )
        )
        claim_records = self.store.claim_versions(claim_ids)
        if self.zenbrain_prior is not None and hasattr(self.zenbrain_prior, "project_claims"):
            claim_projection = self.zenbrain_prior.project_claims(claim_ids)
        else:
            claim_projection = {
                claim_id: {
                    **record,
                    "suppressed": record["status"] in {"rejected", "superseded"},
                    "reason": f"static_{record['status']}" if record["status"] in {"rejected", "superseded"} else None,
                }
                for claim_id, record in claim_records.items()
            }
        if self.zenbrain_prior is not None and candidates:
            try:
                prior_scores = self.zenbrain_prior.score(list(candidates))
                trace["routes"].append("zenbrain_prior")
                trace["zenbrain_prior_candidates"] = len(prior_scores)
                path_ids = [path["path_id"] for path in graph_paths if path.get("path_id")]
                relation_ids = [
                    edge["relation_id"]
                    for path in graph_paths
                    for edge in path.get("path_edges", [])
                    if edge.get("relation_id")
                ]
                if path_ids:
                    path_prior_scores = self.zenbrain_prior.score_targets("path", list(dict.fromkeys(path_ids)))
                if relation_ids:
                    edge_prior_scores = self.zenbrain_prior.score_targets(
                        "relation", list(dict.fromkeys(relation_ids))
                    )
                if claim_ids:
                    claim_prior_scores = self.zenbrain_prior.score_targets("claim", claim_ids)
                for path in graph_paths:
                    path["zenbrain_path_prior"] = float(path_prior_scores.get(path.get("path_id"), 0.0))
                    path["zenbrain_edge_prior"] = max(
                        (
                            float(edge_prior_scores.get(edge.get("relation_id"), 0.0))
                            for edge in path.get("path_edges", [])
                        ),
                        default=0.0,
                    )
                if path_ids or relation_ids:
                    trace["routes"].append("zenbrain_graph_prior")
                    trace["zenbrain_path_candidates"] = len(path_prior_scores)
                    trace["zenbrain_edge_candidates"] = len(edge_prior_scores)
                if claim_ids:
                    trace["zenbrain_claim_candidates"] = len(claim_prior_scores)
            except Exception as exc:  # keep evidence retrieval available
                trace["zenbrain_prior_error"] = str(exc)

        visible_candidates: dict[str, dict[str, Any]] = {}
        suppressed_claim_count = 0
        for item in candidates.values():
            linked_claim_ids = claim_ids_by_observation.get(item["observation_id"], [])
            suppressed_claim_ids = [
                claim_id for claim_id in linked_claim_ids if claim_projection.get(claim_id, {}).get("suppressed")
            ]
            suppressed_claim_count += len(suppressed_claim_ids)
            visible_claim_ids = [claim_id for claim_id in linked_claim_ids if claim_id not in suppressed_claim_ids]
            if linked_claim_ids and not visible_claim_ids:
                continue
            item["zenbrain_prior"] = max(-1.0, min(1.0, float(prior_scores.get(item["observation_id"], 0.0))))
            relevant_paths = [
                path for path in graph_paths if item["observation_id"] in path.get("observation_ids", [])
            ]
            item["zenbrain_path_prior"] = max(
                (float(path.get("zenbrain_path_prior", 0.0)) for path in relevant_paths),
                default=0.0,
            )
            item["zenbrain_edge_prior"] = max(
                (float(path.get("zenbrain_edge_prior", 0.0)) for path in relevant_paths),
                default=0.0,
            )
            item["claim_version_ids"] = visible_claim_ids
            item["suppressed_claim_version_ids"] = suppressed_claim_ids
            item["zenbrain_claim_prior"] = max(
                (float(claim_prior_scores.get(claim_id, 0.0)) for claim_id in visible_claim_ids),
                default=0.0,
            )
            item["claim_conflict_ids"] = []
            route_contributions = {
                route: route_weights[route] / (rrf_k + rank)
                for route, ranks in route_ranks.items()
                if (rank := ranks.get(item["observation_id"])) is not None
            }
            item["route_contributions"] = route_contributions
            item["fusion_score"] = sum(route_contributions.values())
            item["combined_score"] = (
                item["fusion_score"]
                + item["source_quality"] * 0.01
                + item["zenbrain_prior"] * 0.1
                + item["zenbrain_path_prior"] * 0.05
                + item["zenbrain_edge_prior"] * 0.03
                + item["zenbrain_claim_prior"] * 0.05
            )
            visible_candidates[item["observation_id"]] = item
        candidates = visible_candidates
        evidence = self._select_evidence(plan, candidates, limit)
        conflict_versions: dict[str, list[dict[str, Any]]] = {}
        for item in evidence:
            for claim_version_id in item.claim_version_ids:
                record = claim_projection.get(claim_version_id)
                if record and record.get("claim_id"):
                    conflict_versions.setdefault(record["claim_id"], []).append(record)
        conflicts = []
        for claim_id, versions in conflict_versions.items():
            unique_values = {
                (repr(version.get("object_value")), version.get("unit"))
                for version in versions
            }
            if len(unique_values) > 1:
                conflicts.append(
                    {
                        "claim_id": claim_id,
                        "claim_version_ids": [version["claim_version_id"] for version in versions],
                        "values": [
                            {"object_value": version.get("object_value"), "unit": version.get("unit")}
                            for version in versions
                        ],
                        "status": "conflicted",
                    }
                )
        conflict_ids = {conflict["claim_id"] for conflict in conflicts}
        for item in evidence:
            item.claim_conflict_ids = sorted(
                {
                    claim_projection[claim_version_id]["claim_id"]
                    for claim_version_id in item.claim_version_ids
                    if claim_version_id in claim_projection
                    and claim_projection[claim_version_id].get("claim_id") in conflict_ids
                }
            )
        slot_status, slot_evidence = self._evaluate_slots(plan, evidence)
        missing = [name for name, status in slot_status.items() if status is SlotStatus.MISSING]
        ambiguity_question = self._ambiguity_question(query, entity_ids, evidence)
        follow_up_questions = self._follow_up_questions(plan, slot_status, evidence)
        if ambiguity_question is not None:
            follow_up_questions.insert(0, ambiguity_question)
        context_evidence = self._expand_context(
            evidence,
            limit=min(max(limit * 2, 4), 12),
        )
        trace["context_expansion"] = {
            "seed_count": len(evidence),
            "context_count": len(context_evidence),
            "relations": sorted(
                {
                    item.context_relation
                    for item in context_evidence
                    if item.context_relation
                }
            ),
        }
        citations = [
            {
                "role": "primary",
                "observation_id": item.observation_id,
                "asset_id": item.asset_id,
                "element_id": item.element_id,
                "aligned_element_ids": item.aligned_element_ids,
                "locator": item.locator,
                "claim_version_ids": item.claim_version_ids,
                "supporting_citations": item.supporting_citations,
            }
            for item in evidence
        ]
        citations.extend(
            {
                "role": "context",
                "observation_id": item.observation_id,
                "asset_id": item.asset_id,
                "element_id": item.element_id,
                "aligned_element_ids": item.aligned_element_ids,
                "locator": item.locator,
                "context_of_observation_ids": item.context_of_observation_ids,
            }
            for item in context_evidence
        )
        citations.extend(
            {
                **supporting,
                "role": "supporting",
                "supported_by_observation_id": item.observation_id,
            }
            for item in evidence
            for supporting in item.supporting_citations
        )
        plan = plan.model_copy(
            update={
                "routes": trace["routes"],
                "evidence_slots": [
                    slot.model_copy(update={"status": slot_status[slot.name]})
                    for slot in plan.evidence_slots
                ],
            }
        )
        return EvidencePack(
            query=query,
            query_plan=plan,
            slot_status=slot_status,
            slot_evidence=slot_evidence,
            evidence=evidence,
            context_evidence=context_evidence,
            graph_paths=graph_paths,
            missing=missing,
            citations=citations,
            follow_up_required=bool(follow_up_questions),
            follow_up_questions=follow_up_questions,
            retrieval_trace={
                **trace,
                "candidate_count": len(candidates),
                "entity_seed_count": len(entity_ids),
                "graph_path_count": len(graph_paths),
                "graph_max_hops": plan.max_hops,
                "graph_beam_width": plan.beam_width,
                "claim_suppressed_count": suppressed_claim_count,
                "claim_conflict_count": len(conflicts),
            },
            conflicts=conflicts,
        )
