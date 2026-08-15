"""Conservative entity and relation candidates from source observations."""

from __future__ import annotations

import re
from collections import OrderedDict

from .models import Entity, Observation, Relation


# These are aliases observed in the current AAPM material. The extractor is
# deliberately a closed vocabulary until a reviewed entity resolver exists.
KNOWN_ENTITIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("DREME", "method", ("dynamic reconstruction and motion estimation",)),
    ("GeoDose", "method", ("geometry-encoded AI dose calculation framework", "geodose")),
    ("TransFM", "method", ("transfm", "adapted fluence map estimation")),
    ("PTV", "anatomy", ("planning target volume",)),
    ("bladder", "anatomy", ()),
    ("rectum", "anatomy", ()),
)


def _entity_matches(text: str) -> list[tuple[int, int, str, str, tuple[str, ...]]]:
    matches: list[tuple[int, int, str, str, tuple[str, ...]]] = []
    lowered = text.casefold()
    for canonical, entity_type, aliases in KNOWN_ENTITIES:
        names = (canonical, *aliases)
        for name in names:
            start = 0
            needle = name.casefold()
            while True:
                position = lowered.find(needle, start)
                if position < 0:
                    break
                matches.append((position, position + len(name), canonical, entity_type, aliases))
                start = position + max(len(name), 1)
    return matches


def extract_entities(observation: Observation) -> list[Entity]:
    """Extract only reviewed-vocabulary entities from an observation."""
    entities: OrderedDict[str, Entity] = OrderedDict()
    for _, _, canonical, entity_type, aliases in _entity_matches(observation.value):
        entity = Entity.new(
            canonical_name=canonical,
            entity_type=entity_type,
            aliases=aliases,
            extraction_confidence=0.8,
        )
        entities.setdefault(entity.entity_id, entity)
    return list(entities.values())


def extract_cooccurrence_relations(observation: Observation, entities: list[Entity]) -> list[Relation]:
    """Create weak, auditable co-occurrence edges; never label them causal."""
    if len(entities) < 2:
        return []
    relations: list[Relation] = []
    for source in entities:
        for target in entities:
            if source.entity_id >= target.entity_id:
                continue
            relations.append(
                Relation.new(
                    source_entity_id=source.entity_id,
                    target_entity_id=target.entity_id,
                    predicate="co_occurs_in_observation",
                    observation_ids=(observation.observation_id,),
                    confidence=0.35,
                    extraction_method="closed-vocabulary-cooccurrence-v1",
                )
            )
    return relations


_SEMANTIC_RELATION_RULES: tuple[tuple[str, str, float], ...] = (
    ("provides_input_to", r"\bprovid(?:e|es|ed)\b.*\bto\b|\bfeed(?:s|ed)?\b.*\bto\b|\b给\b|\b提供\b", 0.78),
    ("takes_input_from", r"\btake(?:s|n)?\b.*\bfrom\b|\b从\b", 0.74),
    ("derived_from", r"\bderived\s+(?:from|based\s+on)\b|\bbased\s+on\b|\b来源于\b|\b基于\b", 0.80),
    ("uses", r"\buse(?:s|d)?\b|\butili[sz](?:e|es|ed)\b|\b使用\b|\b采用\b", 0.74),
    ("predicts", r"\bpredict(?:s|ed)?\b|\bestimat(?:e|es|ed)\b|\b预测\b|\b估计\b", 0.76),
    ("reconstructs", r"\breconstruct(?:s|ed)?\b|\b重建\b|\b重构\b", 0.76),
)

SEMANTIC_RELATION_PREDICATES = frozenset(rule[0] for rule in _SEMANTIC_RELATION_RULES)


def _entity_mentions(text: str) -> list[tuple[int, int, str]]:
    """Return de-duplicated entity spans in source order."""
    mentions = {(start, end, canonical) for start, end, canonical, *_ in _entity_matches(text)}
    return sorted(mentions, key=lambda item: (item[0], item[1], item[2]))


def extract_semantic_relations(observation: Observation, entities: list[Entity]) -> list[Relation]:
    """Extract only high-signal directional relations from explicit wording.

    A relation is emitted only when two reviewed entities occur in order and the
    text between them matches an explicit relation pattern. Unmatched pairs are
    intentionally left to the weak co-occurrence extractor.
    """
    entity_by_name = {entity.canonical_name: entity for entity in entities}
    mentions = _entity_mentions(observation.value)
    relations: list[Relation] = []
    for index, (left_start, left_end, left_name) in enumerate(mentions):
        source = entity_by_name.get(left_name)
        if source is None:
            continue
        for right_start, right_end, right_name in mentions[index + 1 :]:
            target = entity_by_name.get(right_name)
            if target is None or source.entity_id == target.entity_id:
                continue
            context = observation.value[left_end:right_start].casefold()
            for predicate, pattern, confidence in _SEMANTIC_RELATION_RULES:
                if re.search(pattern, context, re.IGNORECASE):
                    relations.append(
                        Relation.new(
                            source_entity_id=source.entity_id,
                            target_entity_id=target.entity_id,
                            predicate=predicate,
                            observation_ids=(observation.observation_id,),
                            confidence=confidence,
                            extraction_method="explicit-pattern-semantic-v1",
                        )
                    )
                    break
    return relations
