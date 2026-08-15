"""Validated Graph 3.0 phase-0 data contracts.

The models deliberately separate immutable source material from derived
observations and normalized claims. Indexes and graph projections should
reference these contracts rather than replace them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import canonical_json, content_hash, stable_id


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SourceLocator(ContractModel):
    page: int | None = Field(default=None, ge=1)
    slide: int | None = Field(default=None, ge=1)
    sheet: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    shape_id: str | None = None
    cell: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    time_start_ms: int | None = Field(default=None, ge=0)
    time_end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def has_location(self) -> "SourceLocator":
        if not any(value is not None for value in self.model_dump().values()):
            raise ValueError("SourceLocator must contain at least one location field")
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        if self.time_start_ms is not None and self.time_end_ms is not None and self.time_end_ms < self.time_start_ms:
            raise ValueError("time_end_ms must be >= time_start_ms")
        return self


class RawAsset(ContractModel):
    asset_id: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str
    byte_size: int = Field(ge=0)
    source_uri: str | None = None
    storage_uri: str
    acquired_at: datetime | None = None
    source_time: datetime | None = None
    document_version: str | None = None
    dataset: str | None = None
    license: str | None = None

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str,
        storage_uri: str,
        source_uri: str | None = None,
        dataset: str | None = None,
        document_version: str | None = None,
    ) -> "RawAsset":
        digest = content_hash(data)
        return cls(
            asset_id=stable_id("asset", {"content_hash": digest}),
            content_hash=digest,
            media_type=media_type,
            byte_size=len(data),
            source_uri=source_uri,
            storage_uri=storage_uri,
            dataset=dataset,
            document_version=document_version,
        )


class ElementType(StrEnum):
    DOCUMENT = "document"
    SECTION = "section"
    SLIDE = "slide"
    TEXT = "text"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    FIGURE = "figure"
    CHART = "chart"
    AUDIO_SEGMENT = "audio_segment"
    VIDEO_SEGMENT = "video_segment"
    FORMULA = "formula"


class SourceElement(ContractModel):
    element_id: str
    asset_id: str
    element_type: ElementType
    locator: SourceLocator
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str | None = None
    parent_id: str | None = None
    previous_id: str | None = None
    next_id: str | None = None
    duplicate_group_id: str | None = None

    @classmethod
    def from_text(
        cls,
        *,
        asset_id: str,
        element_type: ElementType,
        locator: SourceLocator,
        text: str,
        parent_id: str | None = None,
        duplicate_group_id: str | None = None,
    ) -> "SourceElement":
        digest = content_hash(text)
        element_id = stable_id(
            "element",
            {
                "asset_id": asset_id,
                "locator": locator.model_dump(mode="json"),
                "content_hash": digest,
            },
        )
        return cls(
            element_id=element_id,
            asset_id=asset_id,
            element_type=element_type,
            locator=locator,
            content_hash=digest,
            text=text,
            parent_id=parent_id,
            duplicate_group_id=duplicate_group_id,
        )


class ObservationKind(StrEnum):
    NATIVE_TEXT = "native_text"
    OCR = "ocr"
    ASR = "asr"
    VISION = "vision"
    MANUAL = "manual"
    CONTEXTUAL_PREFIX = "contextual_prefix"


class Observation(ContractModel):
    observation_id: str
    element_id: str
    aligned_element_ids: tuple[str, ...] = ()
    kind: ObservationKind
    value: str
    extractor: str
    extractor_version: str
    prompt_version: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        *,
        element_id: str,
        kind: ObservationKind,
        value: str,
        extractor: str,
        extractor_version: str,
        created_at: datetime,
        prompt_version: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        aligned_element_ids: tuple[str, ...] = (),
    ) -> "Observation":
        all_element_ids = tuple(sorted(set((element_id, *aligned_element_ids))))
        fingerprint = {
            "element_ids": all_element_ids,
            "kind": kind.value,
            "value": value,
            "extractor": extractor,
            "extractor_version": extractor_version,
            "prompt_version": prompt_version,
        }
        return cls(
            observation_id=stable_id("observation", fingerprint),
            element_id=element_id,
            aligned_element_ids=tuple(item for item in all_element_ids if item != element_id),
            kind=kind,
            value=value,
            extractor=extractor,
            extractor_version=extractor_version,
            prompt_version=prompt_version,
            confidence=confidence,
            created_at=created_at,
            metadata=metadata or {},
        )


class ClaimStatus(StrEnum):
    ACTIVE = "active"
    DISPUTED = "disputed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ClaimVersion(ContractModel):
    claim_id: str
    claim_version_id: str
    version: int = Field(ge=1)
    subject: str
    predicate: str
    object_value: Any
    unit: str | None = None
    polarity: Literal["positive", "negative"] = "positive"
    modality: Literal["certain", "possible", "recommended", "hypothetical"] = "certain"
    population: str | None = None
    condition: str | None = None
    method: str | None = None
    valid_time: str | None = None
    observed_time: datetime | None = None
    source_scope: Literal["author_result", "quoted_result", "background"] = "background"
    status: ClaimStatus = ClaimStatus.ACTIVE
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(ge=0.0, le=1.0)
    support_strength: float = Field(ge=0.0, le=1.0)

    @classmethod
    def new(
        cls,
        *,
        subject: str,
        predicate: str,
        object_value: Any,
        extraction_confidence: float,
        source_quality: float,
        support_strength: float,
        version: int = 1,
        **kwargs: Any,
    ) -> "ClaimVersion":
        identity = {
            "subject": subject.strip(),
            "predicate": predicate.strip(),
            "unit": kwargs.get("unit"),
            "population": kwargs.get("population"),
            "condition": kwargs.get("condition"),
            "method": kwargs.get("method"),
            "valid_time": kwargs.get("valid_time"),
        }
        claim_id = stable_id("claim", identity)
        claim_version_id = stable_id(
            "claim_version",
            {
                "claim_id": claim_id,
                "version": version,
                "object_value": object_value,
                "polarity": kwargs.get("polarity", "positive"),
                "modality": kwargs.get("modality", "certain"),
            },
        )
        return cls(
            claim_id=claim_id,
            claim_version_id=claim_version_id,
            version=version,
            subject=subject.strip(),
            predicate=predicate.strip(),
            object_value=object_value,
            extraction_confidence=extraction_confidence,
            source_quality=source_quality,
            support_strength=support_strength,
            **kwargs,
        )


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"
    DERIVED_FROM = "derived_from"
    MENTIONS = "mentions"
    QUOTES = "quotes"


class EvidenceLink(ContractModel):
    link_id: str
    observation_id: str
    claim_version_id: str
    relation: EvidenceRelation
    strength: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None
    created_at: datetime

    @classmethod
    def new(
        cls,
        *,
        observation_id: str,
        claim_version_id: str,
        relation: EvidenceRelation,
        strength: float,
        created_at: datetime,
        rationale: str | None = None,
    ) -> "EvidenceLink":
        fingerprint = {
            "observation_id": observation_id,
            "claim_version_id": claim_version_id,
            "relation": relation.value,
            "strength": strength,
        }
        return cls(
            link_id=stable_id("evidence_link", fingerprint),
            observation_id=observation_id,
            claim_version_id=claim_version_id,
            relation=relation,
            strength=strength,
            rationale=rationale,
            created_at=created_at,
        )


class Entity(ContractModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    source_scope: str = "document"

    @classmethod
    def new(
        cls,
        *,
        canonical_name: str,
        entity_type: str,
        aliases: tuple[str, ...] = (),
        extraction_confidence: float = 0.6,
        source_scope: str = "document",
    ) -> "Entity":
        canonical = " ".join(canonical_name.split())
        entity_id = stable_id(
            "entity",
            {"canonical_name": canonical.casefold(), "entity_type": entity_type.casefold()},
        )
        return cls(
            entity_id=entity_id,
            canonical_name=canonical,
            entity_type=entity_type,
            aliases=tuple(sorted(set(aliases))),
            extraction_confidence=extraction_confidence,
            source_scope=source_scope,
        )


class Relation(ContractModel):
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    predicate: str
    observation_ids: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str

    @classmethod
    def new(
        cls,
        *,
        source_entity_id: str,
        target_entity_id: str,
        predicate: str,
        observation_ids: tuple[str, ...],
        confidence: float,
        extraction_method: str,
    ) -> "Relation":
        canonical_observations = tuple(sorted(set(observation_ids)))
        relation_id = stable_id(
            "relation",
            {
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "predicate": predicate,
            },
        )
        return cls(
            relation_id=relation_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            predicate=predicate,
            observation_ids=canonical_observations,
            confidence=confidence,
            extraction_method=extraction_method,
        )


def contract_json(model: ContractModel) -> str:
    """Stable JSON representation for persistence and test fixtures."""
    return canonical_json(model.model_dump(mode="json"))
