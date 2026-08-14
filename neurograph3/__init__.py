"""NeuroGraph 3.0 data contracts and evidence primitives."""

from .models import (
    ClaimVersion,
    EvidenceLink,
    Observation,
    RawAsset,
    SourceElement,
    SourceLocator,
)
from .retrieval import EvidencePack, FollowUpQuestion, Graph3Retriever, QueryPlan
from .store import Graph3Store
from .extract import extract_numeric_claims
from .entities import extract_cooccurrence_relations, extract_entities, extract_semantic_relations
from .models import Entity, Relation
from .embedding import OpenAICompatibleEmbedder
from .llm_relations import (
    DeepSeekRelationClient,
    RelationExtractionResult,
    RelationProposal,
    RelationRejection,
    validate_relation_proposals,
)
from .pipeline import GraphBuildStats, build_relation_graph, ingest_and_build_graph

__all__ = [
    "ClaimVersion",
    "EvidenceLink",
    "Observation",
    "RawAsset",
    "SourceElement",
    "SourceLocator",
    "EvidencePack",
    "FollowUpQuestion",
    "Graph3Retriever",
    "Graph3Store",
    "QueryPlan",
    "extract_numeric_claims",
    "Entity",
    "Relation",
    "extract_entities",
    "extract_cooccurrence_relations",
    "extract_semantic_relations",
    "OpenAICompatibleEmbedder",
    "DeepSeekRelationClient",
    "RelationExtractionResult",
    "RelationProposal",
    "RelationRejection",
    "validate_relation_proposals",
    "GraphBuildStats",
    "build_relation_graph",
    "ingest_and_build_graph",
]
