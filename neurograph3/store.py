"""SQLite-backed authoritative storage for the Graph 3.0 evidence layer."""

from __future__ import annotations

import math
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .ingest import IngestResult
from .models import ClaimVersion, Entity, EvidenceLink, Relation


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

            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
                target_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
                predicate TEXT NOT NULL,
                observation_ids_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observation_embeddings (
                observation_id TEXT PRIMARY KEY REFERENCES observations(observation_id),
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL
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
            for table in ("raw_assets", "source_elements", "observations", "claim_versions", "evidence_links", "entities", "relations", "observation_embeddings")
        }

    def list_observations(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT observation_id, value FROM observations ORDER BY observation_id"
        ).fetchall()
        return [{"observation_id": row["observation_id"], "value": row["value"]} for row in rows]

    def put_embeddings(self, model: str, embeddings: dict[str, list[float]]) -> None:
        with self.connection:
            for observation_id, vector in embeddings.items():
                self.connection.execute(
                    """
                    INSERT INTO observation_embeddings(observation_id, model, dimensions, vector_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(observation_id) DO UPDATE SET
                        model=excluded.model,
                        dimensions=excluded.dimensions,
                        vector_json=excluded.vector_json
                    """,
                    (observation_id, model, len(vector), json.dumps(vector, separators=(",", ":"))),
                )

    def search_vector(self, query_vector: list[float], limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT e.observation_id, e.vector_json, o.*, s.locator_json,
                   s.asset_id, s.element_type
            FROM observation_embeddings e
            JOIN observations o ON o.observation_id = e.observation_id
            JOIN source_elements s ON s.element_id = o.element_id
            """
        ).fetchall()
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            return []
        hits: list[dict[str, Any]] = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            if len(vector) != len(query_vector):
                continue
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                continue
            score = sum(a * b for a, b in zip(query_vector, vector)) / (query_norm * norm)
            hits.append(self._observation_hit(row, vector_score=score))
        hits.sort(key=lambda item: item["vector_score"], reverse=True)
        return hits[:limit]

    def _observation_hit(self, row: sqlite3.Row, **scores: float) -> dict[str, Any]:
        hit = {
            "observation_id": row["observation_id"],
            "element_id": row["element_id"],
            "aligned_element_ids": json.loads(row["aligned_element_ids_json"]),
            "kind": row["kind"],
            "value": row["value"],
            "asset_id": row["asset_id"],
            "element_type": row["element_type"],
            "locator": json.loads(row["locator_json"]),
            "matched_terms": 0,
            "matched_numbers": 0,
            "lexical_score": 0.0,
        }
        hit.update(scores)
        return hit

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

    def put_graph(self, entities: list[Entity], relations: list[Relation]) -> None:
        """Persist conservative entity/relation candidates idempotently."""
        with self.connection:
            for entity in entities:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO entities(
                        entity_id, canonical_name, entity_type, aliases_json, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entity.entity_id,
                        entity.canonical_name,
                        entity.entity_type,
                        json.dumps(entity.aliases, ensure_ascii=False),
                        json.dumps(entity.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    ),
                )
            for relation in relations:
                existing = self.connection.execute(
                    "SELECT observation_ids_json, confidence FROM relations WHERE relation_id = ?",
                    (relation.relation_id,),
                ).fetchone()
                if existing is None:
                    self.connection.execute(
                        """
                        INSERT INTO relations(
                            relation_id, source_entity_id, target_entity_id, predicate,
                            observation_ids_json, confidence, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            relation.relation_id,
                            relation.source_entity_id,
                            relation.target_entity_id,
                            relation.predicate,
                            json.dumps(relation.observation_ids, ensure_ascii=False),
                            relation.confidence,
                            json.dumps(relation.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                        ),
                    )
                    continue

                observation_ids = tuple(
                    sorted(set(json.loads(existing["observation_ids_json"])) | set(relation.observation_ids))
                )
                confidence = max(float(existing["confidence"]), relation.confidence)
                payload = relation.model_dump(mode="json")
                payload["observation_ids"] = list(observation_ids)
                payload["confidence"] = confidence
                self.connection.execute(
                    """
                    UPDATE relations
                    SET observation_ids_json = ?, confidence = ?, payload_json = ?
                    WHERE relation_id = ?
                    """,
                    (
                        json.dumps(observation_ids, ensure_ascii=False),
                        confidence,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        relation.relation_id,
                    ),
                )

    def search_entities(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        rows = self.connection.execute(
            """
            SELECT entity_id, canonical_name, entity_type, aliases_json
            FROM entities
            ORDER BY canonical_name
            LIMIT ?
            """,
            (max(limit, 100),),
        ).fetchall()
        normalized_query = " ".join(query.casefold().split())
        matches = []
        for row in rows:
            aliases = json.loads(row["aliases_json"])
            names = (row["canonical_name"], *aliases)
            if not any(" ".join(name.casefold().split()) in normalized_query for name in names):
                continue
            matches.append(
                {
                    "entity_id": row["entity_id"],
                    "canonical_name": row["canonical_name"],
                    "entity_type": row["entity_type"],
                    "aliases": aliases,
                }
            )
            if len(matches) >= limit:
                break
        return matches

    def list_entities(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT entity_id, canonical_name, entity_type, aliases_json
            FROM entities
            ORDER BY canonical_name
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "entity_id": row["entity_id"],
                "canonical_name": row["canonical_name"],
                "entity_type": row["entity_type"],
                "aliases": json.loads(row["aliases_json"]),
            }
            for row in rows
        ]

    def observation_hits(self, observation_ids: list[str]) -> list[dict[str, Any]]:
        if not observation_ids:
            return []
        placeholders = ",".join("?" for _ in observation_ids)
        rows = self.connection.execute(
            f"""
            SELECT o.*, s.locator_json, s.asset_id, s.element_type
            FROM observations o
            JOIN source_elements s ON s.element_id = o.element_id
            WHERE o.observation_id IN ({placeholders})
            """,
            observation_ids,
        ).fetchall()
        return [self._observation_hit(row) for row in rows]

    def expand_graph(
        self,
        entity_ids: list[str],
        *,
        max_hops: int = 1,
        allowed_predicates: tuple[str, ...] = (
            "provides_input_to",
            "takes_input_from",
            "derived_from",
            "uses",
            "predicts",
            "reconstructs",
            "co_occurs_in_observation",
        ),
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Expand only typed edges and return paths plus observation hits."""
        if max_hops < 1 or not entity_ids:
            return [], []
        allowed = set(allowed_predicates)
        frontier = set(entity_ids)
        visited = set(frontier)
        paths: list[dict[str, Any]] = []
        observation_ids: set[str] = set()
        for hop in range(1, max_hops + 1):
            rows = self.connection.execute(
                """
                SELECT r.*, se.canonical_name AS source_name, te.canonical_name AS target_name
                FROM relations r
                JOIN entities se ON se.entity_id = r.source_entity_id
                JOIN entities te ON te.entity_id = r.target_entity_id
                """
            ).fetchall()
            next_frontier: set[str] = set()
            for row in rows:
                if row["predicate"] not in allowed:
                    continue
                source_id, target_id = row["source_entity_id"], row["target_entity_id"]
                if source_id not in frontier and target_id not in frontier:
                    continue
                other_id = target_id if source_id in frontier else source_id
                if other_id in visited and other_id not in frontier:
                    continue
                ids = json.loads(row["observation_ids_json"])
                observation_ids.update(ids)
                paths.append(
                    {
                        "hop": hop,
                        "source_entity_id": source_id,
                        "source_name": row["source_name"],
                        "predicate": row["predicate"],
                        "target_entity_id": target_id,
                        "target_name": row["target_name"],
                        "observation_ids": ids,
                        "confidence": row["confidence"],
                        "extraction_method": json.loads(row["payload_json"]).get("extraction_method"),
                    }
                )
                next_frontier.add(other_id)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        return paths, self.observation_hits(sorted(observation_ids))

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
            hit = self._observation_hit(row, matched_terms=matched_terms, matched_numbers=matched_numbers, lexical_score=score)
            scored.append(hit)
        scored.sort(key=lambda item: (item["matched_numbers"], item["lexical_score"]), reverse=True)
        return scored[:limit]
