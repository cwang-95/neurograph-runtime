"""DeepSeek relation proposals with a strict, auditable validation boundary."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from pydantic import Field

from .entities import SEMANTIC_RELATION_PREDICATES
from .models import ContractModel, Entity, Observation, Relation


class RelationProposal(ContractModel):
    source: str
    predicate: str
    target: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RelationRejection(ContractModel):
    proposal: dict[str, Any] = Field(default_factory=dict)
    reason: str


class RelationExtractionResult(ContractModel):
    accepted: list[Relation] = Field(default_factory=list)
    rejected: list[RelationRejection] = Field(default_factory=list)
    model: str = ""


class DeepSeekRelationError(RuntimeError):
    pass


def load_deepseek_api_key(config_path: str | Path | None = None) -> str:
    """Load the existing local key without exposing it in logs or payloads."""
    environment_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if environment_key:
        return environment_key
    path = Path(config_path or os.path.expanduser("~/.openclaw/openclaw.json"))
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        provider = config.get("models", {}).get("providers", {}).get("custom-api-deepseek-com", {})
        return str(provider.get("apiKey", "") or "").strip()
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return ""


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DeepSeekRelationError("DeepSeek relation response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DeepSeekRelationError("DeepSeek relation response must be a JSON object")
    return value


def _proposal_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("proposal must be an object")
    return raw


def validate_relation_proposals(
    payload: dict[str, Any],
    observation: Observation,
    entities: Sequence[Entity],
    *,
    model: str = "deepseek",
    minimum_confidence: float = 0.65,
) -> RelationExtractionResult:
    """Validate model output and convert only safe proposals to graph relations."""
    entity_by_name: dict[str, Entity] = {}
    for entity in entities:
        entity_by_name[entity.canonical_name.casefold()] = entity
        for alias in entity.aliases:
            entity_by_name[alias.casefold()] = entity

    accepted: list[Relation] = []
    rejected: list[RelationRejection] = []
    raw_relations = payload.get("relations", [])
    if not isinstance(raw_relations, list):
        return RelationExtractionResult(
            rejected=[RelationRejection(reason="relations must be a list")],
            model=model,
        )

    observation_text = observation.value.casefold()
    for raw in raw_relations:
        try:
            proposal_data = _proposal_dict(raw)
            proposal = RelationProposal.model_validate(proposal_data)
        except (TypeError, ValueError) as exc:
            rejected.append(RelationRejection(proposal=raw if isinstance(raw, dict) else {}, reason=f"invalid proposal: {exc}"))
            continue

        source = entity_by_name.get(proposal.source.casefold())
        target = entity_by_name.get(proposal.target.casefold())
        if source is None or target is None:
            rejected.append(RelationRejection(proposal=proposal_data, reason="source or target is not a reviewed entity"))
            continue
        if source.entity_id == target.entity_id:
            rejected.append(RelationRejection(proposal=proposal_data, reason="source and target must differ"))
            continue
        if proposal.predicate not in SEMANTIC_RELATION_PREDICATES:
            rejected.append(RelationRejection(proposal=proposal_data, reason="predicate is not in the semantic relation allowlist"))
            continue
        if proposal.confidence < minimum_confidence:
            rejected.append(RelationRejection(proposal=proposal_data, reason="confidence is below the acceptance threshold"))
            continue
        def present(entity: Entity) -> bool:
            return any(
                name.casefold() in observation_text
                for name in (entity.canonical_name, *entity.aliases)
            )

        if not present(source) or not present(target):
            rejected.append(RelationRejection(proposal=proposal_data, reason="entities are not both present in the observation text"))
            continue

        accepted.append(
            Relation.new(
                source_entity_id=source.entity_id,
                target_entity_id=target.entity_id,
                predicate=proposal.predicate,
                observation_ids=(observation.observation_id,),
                confidence=proposal.confidence,
                extraction_method="deepseek-structured-v1",
            )
        )
    return RelationExtractionResult(accepted=accepted, rejected=rejected, model=model)


class DeepSeekRelationClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = "https://api.deepseek.com/v1/chat/completions",
        model: str = "deepseek-v4-flash",
        timeout: float = 60.0,
        disable_thinking: bool = True,
    ):
        self.api_key = (api_key or load_deepseek_api_key()).strip()
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.disable_thinking = disable_thinking

    def _prompt(self, observation: Observation, entities: Sequence[Entity]) -> str:
        names = [entity.canonical_name for entity in entities]
        predicates = sorted(SEMANTIC_RELATION_PREDICATES)
        return (
            "Extract only explicit directional relations from the observation.\n"
            "Return JSON only: {\"relations\":[{\"source\":\"...\",\"predicate\":\"...\","
            "\"target\":\"...\",\"confidence\":0.0,\"rationale\":\"...\"}]}\n"
            f"Reviewed entities (use exact canonical names or aliases): {json.dumps(names, ensure_ascii=False)}\n"
            f"Allowed predicates: {json.dumps(predicates, ensure_ascii=False)}\n"
            "Do not infer a relation from co-occurrence alone. If no explicit relation exists, return an empty list.\n"
            f"Observation: {observation.value}"
        )

    @staticmethod
    def _ssl_context() -> ssl.SSLContext:
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:  # pragma: no cover - depends on deployment environment
            return ssl.create_default_context()

    def extract(self, observation: Observation, entities: Sequence[Entity]) -> RelationExtractionResult:
        if len(entities) < 2:
            return RelationExtractionResult(model=self.model)
        if not self.api_key:
            raise DeepSeekRelationError("DeepSeek API key is not configured")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a conservative scientific relation extractor."},
                {"role": "user", "content": self._prompt(observation, entities)},
            ],
            "temperature": 0,
        }
        if self.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self._ssl_context()) as response:
                response_payload = json.load(response)
            content = response_payload["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - exercised by API integration
            raise DeepSeekRelationError(f"DeepSeek relation request failed: {exc}") from exc
        parsed = _parse_json_object(content)
        return validate_relation_proposals(parsed, observation, entities, model=self.model)
