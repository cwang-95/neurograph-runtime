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
