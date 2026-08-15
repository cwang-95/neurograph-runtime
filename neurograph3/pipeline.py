"""Controlled Graph 3.0 ingestion and relation projection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .entities import extract_cooccurrence_relations, extract_entities, extract_semantic_relations
from .extract import extract_numeric_claims
from .ingest import IngestResult, ingest_markdown
from .llm_relations import DeepSeekRelationClient
from .models import Entity, Observation, Relation
from .store import Graph3Store


@dataclass(frozen=True)
class GraphBuildStats:
    observation_count: int
    claim_candidates: int
    evidence_links_submitted: int
    entity_count: int
    deterministic_relation_candidates: int
    llm_calls: int
    llm_accepted: int
    llm_rejected: int
    relations_submitted: int
    llm_budget_exhausted: bool


def build_relation_graph(
    store: Graph3Store,
    observations: Sequence[Observation],
    *,
    relation_client: Any | None = None,
    use_deepseek: bool = False,
    max_llm_calls: int = 0,
    include_cooccurrence: bool = True,
) -> GraphBuildStats:
    """Build a graph projection with an explicit DeepSeek budget.

    DeepSeek is opt-in. Its accepted results are merged with deterministic
    candidates, while rejected proposals remain visible in the returned stats
    and never reach ``Graph3Store.put_graph``.
    """
    if use_deepseek and relation_client is None:
        raise ValueError("relation_client is required when use_deepseek=True")
    if max_llm_calls < 0:
        raise ValueError("max_llm_calls must be non-negative")

    entities: dict[str, Entity] = {}
    relations: list[Relation] = []
    claims = []
    deterministic_count = 0
    llm_calls = 0
    llm_accepted = 0
    llm_rejected = 0
    budget_exhausted = False

    for observation in observations:
        claims.extend(extract_numeric_claims(observation))
        extracted_entities = extract_entities(observation)
        entities.update({entity.entity_id: entity for entity in extracted_entities})
        deterministic = extract_semantic_relations(observation, extracted_entities)
        relations.extend(deterministic)
        deterministic_count += len(deterministic)
        if include_cooccurrence:
            relations.extend(extract_cooccurrence_relations(observation, extracted_entities))

        if not use_deepseek or len(extracted_entities) < 2:
            continue
        if max_llm_calls and llm_calls >= max_llm_calls:
            budget_exhausted = True
            continue
        output = relation_client.extract(observation, extracted_entities)
        llm_calls += 1
        relations.extend(output.accepted)
        llm_accepted += len(output.accepted)
        llm_rejected += len(output.rejected)

    store.put_claims(claims)
    store.put_graph(list(entities.values()), relations)
    return GraphBuildStats(
        observation_count=len(observations),
        claim_candidates=len(claims),
        evidence_links_submitted=len(claims),
        entity_count=len(entities),
        deterministic_relation_candidates=deterministic_count,
        llm_calls=llm_calls,
        llm_accepted=llm_accepted,
        llm_rejected=llm_rejected,
        relations_submitted=len(relations),
        llm_budget_exhausted=budget_exhausted,
    )


def ingest_and_build_graph(
    source_path: str | Path,
    *,
    storage_root: str | Path,
    dataset: str | None = None,
    relation_client: Any | None = None,
    use_deepseek: bool = False,
    max_llm_calls: int = 0,
    include_cooccurrence: bool = True,
) -> tuple[IngestResult, GraphBuildStats]:
    result = ingest_markdown(source_path, storage_root=storage_root, dataset=dataset)
    with Graph3Store(storage_root) as store:
        store.put_ingest_result(result)
        stats = build_relation_graph(
            store,
            result.observations,
            relation_client=relation_client,
            use_deepseek=use_deepseek,
            max_llm_calls=max_llm_calls,
            include_cooccurrence=include_cooccurrence,
        )
    return result, stats


def build_deepseek_relation_client(model: str, timeout: float) -> DeepSeekRelationClient:
    return DeepSeekRelationClient(model=model, timeout=timeout)
