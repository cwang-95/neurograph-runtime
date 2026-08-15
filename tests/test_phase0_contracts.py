from datetime import datetime, timezone
import unittest

from neurograph3 import (
    ClaimVersion,
    EvidenceLink,
    Observation,
    RawAsset,
    SourceElement,
    SourceLocator,
)
from neurograph3.models import ElementType, EvidenceRelation, ObservationKind
from neurograph3.ids import content_hash


NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)


class Phase0ContractTests(unittest.TestCase):
    def test_raw_asset_id_is_content_addressed(self):
        first = RawAsset.from_bytes(
            b"same source",
            media_type="text/plain",
            storage_uri="cas://asset-1",
        )
        second = RawAsset.from_bytes(
            b"same source",
            media_type="text/plain",
            storage_uri="cas://different-location",
        )
        self.assertEqual(first.asset_id, second.asset_id)
        self.assertEqual(first.content_hash, content_hash(b"same source"))
        self.assertNotEqual(first.storage_uri, second.storage_uri)


    def test_source_element_id_is_stable_and_location_sensitive(self):
        asset = RawAsset.from_bytes(b"slide deck", media_type="application/vnd.ms-powerpoint", storage_uri="cas://deck")
        locator = SourceLocator(slide=5, shape_id="shape-7")
        first = SourceElement.from_text(
            asset_id=asset.asset_id,
            element_type=ElementType.TEXT,
            locator=locator,
            text="GeoDose: 70.1 ms",
        )
        second = SourceElement.from_text(
            asset_id=asset.asset_id,
            element_type=ElementType.TEXT,
            locator=locator,
            text="GeoDose: 70.1 ms",
        )
        other_location = SourceElement.from_text(
            asset_id=asset.asset_id,
            element_type=ElementType.TEXT,
            locator=SourceLocator(slide=6, shape_id="shape-7"),
            text="GeoDose: 70.1 ms",
        )
        self.assertEqual(first.element_id, second.element_id)
        self.assertNotEqual(first.element_id, other_location.element_id)


    def test_observation_deduplicates_same_extraction(self):
        first = Observation.from_value(
            element_id="element_1",
            kind=ObservationKind.ASR,
            value="real-time dose accumulation",
            extractor="whisper",
            extractor_version="1",
            created_at=NOW,
        )
        second = Observation.from_value(
            element_id="element_1",
            kind=ObservationKind.ASR,
            value="real-time dose accumulation",
            extractor="whisper",
            extractor_version="1",
            created_at=NOW,
        )
        self.assertEqual(first.observation_id, second.observation_id)


    def test_claim_versions_keep_conflicting_values_under_one_logical_claim(self):
        first = ClaimVersion.new(
            subject="GeoDose",
            predicate="runtime",
            object_value=70.1,
            unit="ms",
            source_scope="author_result",
            extraction_confidence=0.95,
            source_quality=0.9,
            support_strength=0.95,
        )
        second = ClaimVersion.new(
            subject="GeoDose",
            predicate="runtime",
            object_value=70.1,
            unit="ms",
            source_scope="author_result",
            extraction_confidence=0.95,
            source_quality=0.9,
            support_strength=0.95,
            version=2,
        )
        self.assertEqual(first.claim_id, second.claim_id)
        self.assertNotEqual(first.claim_version_id, second.claim_version_id)


    def test_evidence_link_is_deterministic(self):
        first = EvidenceLink.new(
            observation_id="observation_1",
            claim_version_id="claim_version_1",
            relation=EvidenceRelation.SUPPORTS,
            strength=0.9,
            created_at=NOW,
        )
        second = EvidenceLink.new(
            observation_id="observation_1",
            claim_version_id="claim_version_1",
            relation=EvidenceRelation.SUPPORTS,
            strength=0.9,
            created_at=NOW,
        )
        self.assertEqual(first.link_id, second.link_id)


    def test_locator_rejects_reversed_time_range(self):
        with self.assertRaises(ValueError):
            SourceLocator(time_start_ms=100, time_end_ms=10)


if __name__ == "__main__":
    unittest.main()
