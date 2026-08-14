"""SQLite-backed authoritative storage for the Graph 3.0 evidence layer."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .ingest import IngestResult
from .models import ClaimVersion, EvidenceLink


class Graph3Store:
    """Small, transactional store used before vector/graph projections exist."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "metadata.sqlite3"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Graph3Store":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS raw_assets (
                asset_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_elements (
                element_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL REFERENCES raw_assets(asset_id),
                element_type TEXT NOT NULL,
                locator_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                text TEXT,
                parent_id TEXT,
                previous_id TEXT,
                next_id TEXT,
                duplicate_group_id TEXT
            );

            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                element_id TEXT NOT NULL REFERENCES source_elements(element_id),
                aligned_element_ids_json TEXT NOT NULL,
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                extractor TEXT NOT NULL,
                extractor_version TEXT NOT NULL,
                prompt_version TEXT,
                confidence REAL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claim_versions (
                claim_version_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_value_json TEXT NOT NULL,
                unit TEXT,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evidence_links (
                link_id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL REFERENCES observations(observation_id),
                claim_version_id TEXT NOT NULL REFERENCES claim_versions(claim_version_id),
                relation TEXT NOT NULL,
                strength REAL NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS observation_fts USING fts5(
                observation_id UNINDEXED,
                value,
                tokenize = 'unicode61'
            );

            CREATE INDEX IF NOT EXISTS idx_elements_asset ON source_elements(asset_id);
            CREATE INDEX IF NOT EXISTS idx_elements_duplicate ON source_elements(duplicate_group_id);
            CREATE INDEX IF NOT EXISTS idx_observations_element ON observations(element_id);
            """
        )
        self.connection.commit()

    def put_ingest_result(self, result: IngestResult) -> None:
        """Persist an ingest result idempotently in one transaction."""
        conn = self.connection
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO raw_assets(asset_id, content_hash, payload_json) VALUES (?, ?, ?)",
                (
                    result.asset.asset_id,
                    result.asset.content_hash,
                    json.dumps(result.asset.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                ),
            )
            for element in result.elements:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO source_elements(
                        element_id, asset_id, element_type, locator_json, content_hash,
                        text, parent_id, previous_id, next_id, duplicate_group_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        element.element_id,
                        element.asset_id,
                        element.element_type.value,
                        json.dumps(element.locator.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                        element.content_hash,
                        element.text,
                        element.parent_id,
                        element.previous_id,
                        element.next_id,
                        element.duplicate_group_id,
                    ),
                )
            for observation in result.observations:
                inserted = conn.execute(
                    """
                    INSERT OR IGNORE INTO observations(
                        observation_id, element_id, aligned_element_ids_json, kind,
                        value, extractor, extractor_version, prompt_version, confidence,
                        created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.observation_id,
                        observation.element_id,
                        json.dumps(observation.aligned_element_ids, ensure_ascii=False),
                        observation.kind.value,
                        observation.value,
                        observation.extractor,
                        observation.extractor_version,
                        observation.prompt_version,
                        observation.confidence,
                        observation.created_at.isoformat(),
                        json.dumps(observation.metadata, ensure_ascii=False, sort_keys=True),
                    ),
                )
                if inserted.rowcount:
                    conn.execute(
                        "INSERT OR IGNORE INTO observation_fts(observation_id, value) VALUES (?, ?)",
                        (observation.observation_id, observation.value),
                    )

    def counts(self) -> dict[str, int]:
        return {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("raw_assets", "source_elements", "observations", "claim_versions", "evidence_links")
        }

    def put_claims(self, claims: list[tuple[ClaimVersion, EvidenceLink]]) -> None:
        """Persist candidate claims and their observation links idempotently."""
        with self.connection:
            for claim, link in claims:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO claim_versions(
                        claim_version_id, claim_id, subject, predicate,
                        object_value_json, unit, status, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim.claim_version_id,
                        claim.claim_id,
                        claim.subject,
                        claim.predicate,
                        json.dumps(claim.object_value, ensure_ascii=False, sort_keys=True),
                        claim.unit,
                        claim.status.value,
                        json.dumps(claim.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    ),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO evidence_links(
                        link_id, observation_id, claim_version_id,
                        relation, strength, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        link.link_id,
                        link.observation_id,
                        link.claim_version_id,
                        link.relation.value,
                        link.strength,
                        json.dumps(link.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    ),
                )

    def search_lexical(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search exact terms and numbers, returning source-aware hits."""
        if limit < 1:
            return []
        tokens = re.findall(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]+", query.casefold())
        numeric_tokens = re.findall(r"\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?", query)
        rows: list[sqlite3.Row] = []
        if tokens:
            fts_query = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)
            try:
                rows = list(
                    self.connection.execute(
                        """
                        SELECT o.*, e.locator_json, e.asset_id, e.element_type
                        FROM observation_fts f
                        JOIN observations o ON o.observation_id = f.observation_id
                        JOIN source_elements e ON e.element_id = o.element_id
                        WHERE observation_fts MATCH ?
                        LIMIT ?
                        """,
                        (fts_query, max(limit * 4, limit)),
                    )
                )
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            like_terms = [f"%{token}%" for token in tokens if len(token) > 1]
            if like_terms:
                clauses = " OR ".join("lower(o.value) LIKE ?" for _ in like_terms)
                rows = list(
                    self.connection.execute(
                        f"""
                        SELECT o.*, e.locator_json, e.asset_id, e.element_type
                        FROM observations o
                        JOIN source_elements e ON e.element_id = o.element_id
                        WHERE {clauses}
                        LIMIT ?
                        """,
                        (*like_terms, max(limit * 4, limit)),
                    )
                )

        scored: list[dict[str, Any]] = []
        for row in rows:
            value = row["value"]
            lowered = value.casefold()
            matched_terms = sum(token in lowered for token in tokens if token)
            matched_numbers = sum(number.casefold() in lowered for number in numeric_tokens)
            score = float(matched_terms) + float(matched_numbers * 2)
            scored.append(
                {
                    "observation_id": row["observation_id"],
                    "element_id": row["element_id"],
                    "aligned_element_ids": json.loads(row["aligned_element_ids_json"]),
                    "kind": row["kind"],
                    "value": value,
                    "asset_id": row["asset_id"],
                    "element_type": row["element_type"],
                    "locator": json.loads(row["locator_json"]),
                    "matched_terms": matched_terms,
                    "matched_numbers": matched_numbers,
                    "lexical_score": score,
                }
            )
        scored.sort(key=lambda item: (item["matched_numbers"], item["lexical_score"]), reverse=True)
        return scored[:limit]
