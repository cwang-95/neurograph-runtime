"""Deterministic answer assembly from a Graph 3.0 EvidencePack."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .retrieval import EvidencePack, EvidenceItem


class AnswerCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    observation_id: str
    asset_id: str
    element_id: str
    locator: dict[str, Any]


class AnswerEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    citation_id: str
    observation_id: str
    role: Literal["primary", "context"] = "primary"


class AnswerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    title: str
    evidence: list[AnswerEvidence] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    """A safe answer package for Codex/OpenClaw or an optional final model."""

    model_config = ConfigDict(extra="forbid")

    query: str
    status: Literal["answer", "follow_up", "conflict"]
    response_markdown: str
    sections: list[AnswerSection] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    follow_up_questions: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    used_observation_ids: list[str] = Field(default_factory=list)


def _compact(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _locator(item: EvidenceItem) -> dict[str, Any]:
    if hasattr(item.locator, "model_dump"):
        return item.locator.model_dump(mode="json")
    return dict(item.locator)


def _locator_text(locator: dict[str, Any]) -> str:
    parts: list[str] = []
    if locator.get("slide") is not None:
        parts.append(f"slide {locator['slide']}")
    if locator.get("page") is not None:
        parts.append(f"page {locator['page']}")
    if locator.get("char_start") is not None and locator.get("char_end") is not None:
        parts.append(f"chars {locator['char_start']}-{locator['char_end']}")
    return ", ".join(parts) or "source locator unavailable"


def _kind_for(item: EvidenceItem, pack: EvidencePack) -> str:
    matched = [
        slot
        for slot, observation_ids in pack.slot_evidence.items()
        if item.observation_id in observation_ids
    ]
    if "quantitative_result" in matched:
        return "quantitative_result"
    if "mechanism" in matched:
        return "mechanism"
    return "direct_evidence"


def _title(kind: str) -> str:
    return {
        "direct_evidence": "直接证据",
        "mechanism": "机制与流程",
        "quantitative_result": "量化结果",
        "comparison": "比较结果",
        "limitations": "限制与适用范围",
    }.get(kind, kind)


def _citation(item: EvidenceItem, citation_id: str) -> AnswerCitation:
    return AnswerCitation(
        citation_id=citation_id,
        observation_id=item.observation_id,
        asset_id=item.asset_id,
        element_id=item.element_id,
        locator=_locator(item),
    )


def _select_items(pack: EvidencePack, max_items: int, max_chars: int) -> list[tuple[EvidenceItem, str]]:
    selected: list[tuple[EvidenceItem, str]] = []
    seen_values: set[str] = set()
    for item in pack.evidence:
        value = _compact(item.value, max_chars)
        normalized = re.sub(r"\s+", " ", value).casefold()
        if not value or normalized in seen_values:
            continue
        seen_values.add(normalized)
        selected.append((item, value))
        if len(selected) >= max_items:
            break
    return selected


def assemble_answer(
    pack: EvidencePack,
    *,
    max_items: int = 8,
    max_chars_per_item: int = 900,
) -> AnswerDraft:
    """Assemble a citation-preserving answer without inventing facts."""

    selected = _select_items(pack, max_items, max_chars_per_item)
    citations: list[AnswerCitation] = []
    sections_by_kind: dict[str, AnswerSection] = {}
    answer_items: list[AnswerEvidence] = []
    for index, (item, text) in enumerate(selected, start=1):
        citation_id = f"cite-{index}"
        citations.append(_citation(item, citation_id))
        evidence = AnswerEvidence(
            text=text,
            citation_id=citation_id,
            observation_id=item.observation_id,
        )
        kind = _kind_for(item, pack)
        section = sections_by_kind.setdefault(
            kind,
            AnswerSection(kind=kind, title=_title(kind)),
        )
        section.evidence.append(evidence)
        answer_items.append(evidence)

    sections = list(sections_by_kind.values())
    follow_ups = [question.model_dump(mode="json") for question in pack.follow_up_questions]
    if pack.conflicts:
        status: Literal["answer", "follow_up", "conflict"] = "conflict"
        response_lines = [
            "当前证据对同一事实存在未裁决冲突，暂不合并成单一结论。",
            "请确认采用哪一份来源，或补充时间范围/版本信息。",
        ]
    elif pack.follow_up_required:
        status = "follow_up"
        response_lines = []
        if answer_items:
            response_lines.append("当前可以确认的证据如下，但还不足以安全完成回答：")
            response_lines.extend(
                f"- {item.text} [{item.citation_id}]" for item in answer_items
            )
        if follow_ups:
            if response_lines:
                response_lines.append("")
            response_lines.append(f"需要进一步确认：{follow_ups[0]['question']}")
    else:
        status = "answer"
        response_lines = ["根据当前可核验证据："]
        for section in sections:
            response_lines.append(f"\n### {section.title}")
            response_lines.extend(
                f"- {item.text} [{item.citation_id}]" for item in section.evidence
            )

    return AnswerDraft(
        query=pack.query,
        status=status,
        response_markdown="\n".join(response_lines),
        sections=sections,
        citations=citations,
        missing_slots=pack.missing,
        follow_up_questions=follow_ups,
        conflicts=pack.conflicts,
        used_observation_ids=[item.observation_id for item, _ in selected],
    )


__all__ = [
    "AnswerCitation",
    "AnswerDraft",
    "AnswerEvidence",
    "AnswerSection",
    "assemble_answer",
]
