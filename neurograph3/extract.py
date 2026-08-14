"""Conservative candidate Claim extraction from observations.

This is intentionally deterministic and low-confidence. It extracts numeric
claims for indexing and review; it does not decide that a claim is clinically
or scientifically correct.
"""

from __future__ import annotations

import re
from datetime import datetime

from .models import ClaimVersion, EvidenceLink, EvidenceRelation, Observation


_NUMBER_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|milliseconds?|s|seconds?|Gy|gy|cc|%|percent)\b",
    re.IGNORECASE,
)
_SUBJECTS = ("DREME", "GeoDose", "TransFM", "PTV", "bladder", "rectum")


def _subject_for(text: str, position: int) -> str:
    before = text[:position]
    matches = [(before.casefold().rfind(subject.casefold()), subject) for subject in _SUBJECTS]
    matches = [(index, subject) for index, subject in matches if index >= 0]
    return max(matches, default=(-1, "document"))[1]


def _predicate_for(context: str, unit: str) -> str:
    lowered = context.casefold()
    if unit.casefold() in {"ms", "milliseconds", "s", "seconds"} or any(
        word in lowered for word in ("耗时", "runtime", "time", "计算")
    ):
        return "runtime"
    if "误差" in context or "error" in lowered:
        return "relative_error"
    if "gamma" in lowered or "通过率" in context:
        return "gamma_pass_rate"
    if unit.casefold() in {"gy", "cc"}:
        return "dose_or_volume"
    return "numeric_value"


def extract_numeric_claims(
    observation: Observation,
    *,
    created_at: datetime | None = None,
    source_quality: float = 0.5,
) -> list[tuple[ClaimVersion, EvidenceLink]]:
    """Extract numeric candidate claims and bind each to its observation."""
    extracted: list[tuple[ClaimVersion, EvidenceLink]] = []
    for match in _NUMBER_RE.finditer(observation.value):
        raw_unit = match.group("unit")
        unit = {
            "milliseconds": "ms",
            "millisecond": "ms",
            "seconds": "s",
            "second": "s",
            "percent": "%",
        }.get(raw_unit.casefold(), raw_unit)
        context_start = max(0, match.start() - 120)
        context = observation.value[context_start : match.end()]
        claim = ClaimVersion.new(
            subject=_subject_for(observation.value, match.start()),
            predicate=_predicate_for(context, unit),
            object_value=float(match.group("value")),
            unit=unit,
            source_scope="background",
            observed_time=created_at,
            extraction_confidence=0.65,
            source_quality=source_quality,
            support_strength=0.55,
        )
        link = EvidenceLink.new(
            observation_id=observation.observation_id,
            claim_version_id=claim.claim_version_id,
            relation=EvidenceRelation.MENTIONS,
            strength=0.55,
            rationale="deterministic numeric candidate extraction",
            created_at=created_at or observation.created_at,
        )
        extracted.append((claim, link))
    return extracted
