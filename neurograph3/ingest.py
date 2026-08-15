"""Phase-1 source ingestion for Markdown exports of talks and documents."""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .ids import content_hash, stable_id
from .models import (
    ElementType,
    Observation,
    ObservationKind,
    RawAsset,
    SourceElement,
    SourceLocator,
)


@dataclass(frozen=True)
class IngestResult:
    asset: RawAsset
    elements: tuple[SourceElement, ...]
    observations: tuple[Observation, ...]


_SLIDE_RE = re.compile(r"^##\s+Slide\s+(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
_SECTION_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_RE = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.DOTALL)


def _section_kind(title: str) -> tuple[ElementType, ObservationKind] | None:
    normalized = title.strip().lower()
    if "讲者语音" in title or "speech" in normalized or "transcript" in normalized:
        return ElementType.AUDIO_SEGMENT, ObservationKind.ASR
    if "ppt" in normalized or "视觉" in title or "visual" in normalized:
        return ElementType.FIGURE, ObservationKind.VISION
    return ElementType.TEXT, ObservationKind.NATIVE_TEXT


def _normalize_for_group(value: str) -> str:
    return " ".join(value.casefold().split()).strip()


def _content_pieces(value: str, kind: ObservationKind) -> list[tuple[int, str]]:
    """Split ASR into locatable sentence pieces; keep other modalities intact."""
    if kind is not ObservationKind.ASR:
        return [(0, value)]
    pieces = [(match.start(), match.group(0).strip()) for match in _SENTENCE_RE.finditer(value)]
    return [(offset, piece) for offset, piece in pieces if piece]


def _copy_to_content_store(data: bytes, source_path: Path, storage_root: Path) -> Path:
    digest = content_hash(data)
    suffix = source_path.suffix.lower() or ".bin"
    destination = storage_root / "assets" / digest[:2] / f"{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source_path, destination)
    return destination


def ingest_markdown(
    source_path: str | Path,
    *,
    storage_root: str | Path,
    dataset: str | None = None,
    extractor_version: str = "markdown-structure-v1",
    created_at: datetime | None = None,
) -> IngestResult:
    """Ingest a Markdown document while preserving slide and section locations.

    Exact repeated section content is represented by separate SourceElements
    (each occurrence remains locatable) but one Observation with aligned
    element IDs, preventing repeated source text from multiplying downstream
    Claim evidence.
    """
    source = Path(source_path).expanduser().resolve()
    data = source.read_bytes()
    storage = _copy_to_content_store(data, source, Path(storage_root).expanduser().resolve())
    timestamp = created_at or datetime.now(timezone.utc)
    asset = RawAsset.from_bytes(
        data,
        media_type="text/markdown",
        storage_uri=str(storage),
        source_uri=str(source),
        dataset=dataset,
    )

    document_element = SourceElement.from_text(
        asset_id=asset.asset_id,
        element_type=ElementType.DOCUMENT,
        locator=SourceLocator(char_start=0, char_end=len(data.decode("utf-8", errors="replace"))),
        text=data.decode("utf-8", errors="replace"),
    )
    elements: list[SourceElement] = [document_element]
    occurrence_groups: dict[tuple[ObservationKind, str], list[SourceElement]] = defaultdict(list)
    previous_slide_element: SourceElement | None = None

    slide_matches = list(_SLIDE_RE.finditer(document_element.text or ""))
    if not slide_matches:
        slide_matches = [None]

    for index, slide_match in enumerate(slide_matches):
        if slide_match is None:
            slide_number = None
            body = document_element.text or ""
            body_offset = 0
        else:
            slide_number = int(slide_match.group(1))
            body_start = slide_match.end()
            body_end = slide_matches[index + 1].start() if index + 1 < len(slide_matches) else len(document_element.text or "")
            body = (document_element.text or "")[body_start:body_end].strip()
            body_offset = body_start
        slide_element = SourceElement.from_text(
            asset_id=asset.asset_id,
            element_type=ElementType.SLIDE,
            locator=SourceLocator(slide=slide_number or 1, char_start=body_offset, char_end=body_offset + len(body)),
            text=body,
            parent_id=document_element.element_id,
        )
        if previous_slide_element is not None:
            previous_slide_element.next_id = slide_element.element_id
            slide_element.previous_id = previous_slide_element.element_id
        previous_slide_element = slide_element
        elements.append(slide_element)

        section_matches = list(_SECTION_RE.finditer(body))
        if not section_matches:
            section_matches = [None]
        for section_index, section_match in enumerate(section_matches):
            if section_match is None:
                title = "document_body"
                section_text = body
                section_offset = body_offset
                kind_pair = (ElementType.TEXT, ObservationKind.NATIVE_TEXT)
            else:
                title = section_match.group(1).strip()
                section_start = section_match.end()
                section_end = section_matches[section_index + 1].start() if section_index + 1 < len(section_matches) else len(body)
                section_text = body[section_start:section_end].strip()
                section_offset = body_offset + section_start
                kind_pair = _section_kind(title) or (ElementType.TEXT, ObservationKind.NATIVE_TEXT)
            if not section_text:
                continue
            for piece_offset, piece_text in _content_pieces(section_text, kind_pair[1]):
                element = SourceElement.from_text(
                    asset_id=asset.asset_id,
                    element_type=kind_pair[0],
                    locator=SourceLocator(
                        slide=slide_number or 1,
                        char_start=section_offset + piece_offset,
                        char_end=section_offset + piece_offset + len(piece_text),
                    ),
                    text=piece_text,
                    parent_id=slide_element.element_id,
                    duplicate_group_id=stable_id(
                        "duplicate_group",
                        {"kind": kind_pair[1].value, "text": _normalize_for_group(piece_text)},
                    ),
                )
                elements.append(element)
                occurrence_groups[(kind_pair[1], _normalize_for_group(piece_text))].append(element)

    observations_by_group: list[Observation] = []
    for (kind, _normalized_value), occurrences in occurrence_groups.items():
        first, *aligned = occurrences
        observations_by_group.append(
            Observation.from_value(
                element_id=first.element_id,
                aligned_element_ids=tuple(item.element_id for item in aligned),
                kind=kind,
                value=first.text or "",
                extractor="markdown-structure",
                extractor_version=extractor_version,
                created_at=timestamp,
                metadata={
                    "source_asset_id": asset.asset_id,
                    "occurrence_count": len(occurrences),
                    "slide_numbers": [item.locator.slide for item in occurrences],
                },
            )
        )

    return IngestResult(asset=asset, elements=tuple(elements), observations=tuple(observations_by_group))
