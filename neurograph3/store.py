"""SQLite-backed authoritative storage for the Graph 3.0 evidence layer."""

from __future__ import annotations

import math
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ids import stable_id
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

            CREATE TABLE IF NOT EXISTS zenbrain_events (
                event_id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                observation_id TEXT REFERENCES observations(observation_id),
                event_type TEXT NOT NULL,
                query TEXT,
                caller TEXT,
                path_id TEXT,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS zenbrain_nodes (
                node_id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                scheduler_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(target_type, target_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS observation_fts USING fts5(
                observation_id UNINDEXED,
                value,
                tokenize = 'unicode61'
            );

            CREATE INDEX IF NOT EXISTS idx_elements_asset ON source_elements(asset_id);
            CREATE INDEX IF NOT EXISTS idx_elements_duplicate ON source_elements(duplicate_group_id);
            CREATE INDEX IF NOT EXISTS idx_observations_element ON observations(element_id);
            CREATE INDEX IF NOT EXISTS idx_zenbrain_events_target ON zenbrain_events(target_type, target_id, created_at);
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
            for table in ("raw_assets", "source_elements", "observations", "claim_versions", "evidence_links", "entities", "relations", "observation_embeddings", "zenbrain_events", "zenbrain_nodes")
        }

    def record_zenbrain_event(
        self,
        *,
        target_type: str,
        target_id: str,
        event_type: str,
        observation_id: str | None = None,
        query: str | None = None,
        caller: str | None = None,
        path_id: str | None = None,
        created_at: datetime | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        occurred_at = created_at or datetime.now(timezone.utc)
        event_payload = payload or {}
        event_id = self.zenbrain_event_id(
            target_type=target_type,
            target_id=target_id,
            event_type=event_type,
            query=query,
            caller=caller,
            path_id=path_id,
            created_at=occurred_at,
            payload=event_payload,
        )
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO zenbrain_events(
                    event_id, target_type, target_id, observation_id, event_type,
                    query, caller, path_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    target_type,
                    target_id,
                    observation_id,
                    event_type,
                    query,
                    caller,
                    path_id,
                    occurred_at.isoformat(),
                    json.dumps(event_payload, ensure_ascii=False, sort_keys=True),
                ),
            )
        return event_id

    def zenbrain_event_id(
        self,
        *,
        target_type: str,
        target_id: str,
        event_type: str,
        query: str | None = None,
        caller: str | None = None,
        path_id: str | None = None,
        created_at: datetime | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        occurred_at = created_at or datetime.now(timezone.utc)
        feedback_id = (payload or {}).get("feedback_id")
        return stable_id(
            "zenbrain_event",
            {
                "target_type": target_type,
                "target_id": target_id,
                "event_type": event_type,
                "query": query,
                "caller": caller,
                "path_id": path_id,
                "feedback_id": feedback_id,
                "created_at": None if feedback_id else occurred_at.isoformat(),
            },
        )

    def zenbrain_event_exists(self, event_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM zenbrain_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return row is not None

    def zenbrain_feedback_counts(self, feedback_id: str) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT target_type, payload_json FROM zenbrain_events"
        ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("feedback_id") == feedback_id:
                target_type = row["target_type"]
                counts[target_type] = counts.get(target_type, 0) + 1
        return counts

    def zenbrain_event_history(
        self,
        target_ids: list[str],
        *,
        target_type: str = "observation",
    ) -> list[dict[str, Any]]:
        if not target_ids:
            return []
        placeholders = ",".join("?" for _ in target_ids)
        rows = self.connection.execute(
            f"""
            SELECT target_id, event_type, created_at, query, caller, path_id
            FROM zenbrain_events
            WHERE target_type = ? AND target_id IN ({placeholders})
            ORDER BY created_at
            """,
            (target_type, *target_ids),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_zenbrain_scheduler(self, target_type: str, target_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT scheduler_json FROM zenbrain_nodes WHERE target_type = ? AND target_id = ?",
            (target_type, target_id),
        ).fetchone()
        return json.loads(row["scheduler_json"]) if row else None

    def put_zenbrain_scheduler(
        self,
        *,
        target_type: str,
        target_id: str,
        scheduler: dict[str, Any],
        created_at: datetime | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        created = (created_at or datetime.now(timezone.utc)).isoformat()
        node_id = stable_id("zenbrain_node", {"target_type": target_type, "target_id": target_id})
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO zenbrain_nodes(
                    node_id, target_type, target_id, scheduler_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_type, target_id) DO UPDATE SET
                    scheduler_json=excluded.scheduler_json,
                    updated_at=excluded.updated_at
                """,
                (
                    node_id,
                    target_type,
                    target_id,
                    json.dumps(scheduler, ensure_ascii=False, sort_keys=True),
                    created,
                    now,
                ),
            )

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

    def claim_version_ids_for_observations(self, observation_ids: list[str]) -> dict[str, list[str]]:
        """Return the claim versions explicitly linked to each observation."""
        if not observation_ids:
            return {}
        placeholders = ",".join("?" for _ in observation_ids)
        rows = self.connection.execute(
            f"""
            SELECT observation_id, claim_version_id
            FROM evidence_links
            WHERE observation_id IN ({placeholders})
            ORDER BY observation_id, claim_version_id
            """,
            observation_ids,
        ).fetchall()
        result = {observation_id: [] for observation_id in observation_ids}
        for row in rows:
            result.setdefault(row["observation_id"], []).append(row["claim_version_id"])
        return result

    def observation_ids_for_claim_versions(self, claim_version_ids: list[str]) -> dict[str, list[str]]:
        """Return source observations for selected claim versions."""
        if not claim_version_ids:
            return {}
        placeholders = ",".join("?" for _ in claim_version_ids)
        rows = self.connection.execute(
            f"""
            SELECT claim_version_id, observation_id
            FROM evidence_links
            WHERE claim_version_id IN ({placeholders})
            ORDER BY claim_version_id, observation_id
            """,
            claim_version_ids,
        ).fetchall()
        result = {claim_version_id: [] for claim_version_id in claim_version_ids}
        for row in rows:
            result.setdefault(row["claim_version_id"], []).append(row["observation_id"])
        return result

    def claim_versions(self, claim_version_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return compact, answer-layer-safe ClaimVersion projections."""
        if not claim_version_ids:
            return {}
        placeholders = ",".join("?" for _ in claim_version_ids)
        rows = self.connection.execute(
            f"""
            SELECT claim_version_id, claim_id, subject, predicate,
                   object_value_json, unit, status, payload_json
            FROM claim_versions
            WHERE claim_version_id IN ({placeholders})
            ORDER BY claim_id, claim_version_id
            """,
            claim_version_ids,
        ).fetchall()
        return {
            row["claim_version_id"]: {
                "claim_version_id": row["claim_version_id"],
                "claim_id": row["claim_id"],
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object_value": json.loads(row["object_value_json"]),
                "unit": row["unit"],
                "version": int(json.loads(row["payload_json"]).get("version", 1)),
                "status": row["status"],
            }
            for row in rows
        }

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

    def put_relation_extraction_results(self, entities: list[Entity], results: list[Any]) -> int:
        """Persist only validated relation results; rejected proposals never enter the graph."""
        accepted = [relation for result in results for relation in result.accepted]
        self.put_graph(entities, accepted)
        return len(accepted)

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
        beam_width: int = 20,
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
        """Expand typed edges with bounded multi-hop beam search."""
        if max_hops < 1 or beam_width < 1 or not entity_ids:
            return [], []
        allowed = set(allowed_predicates)
        rows = self.connection.execute(
            """
            SELECT r.*, se.canonical_name AS source_name, te.canonical_name AS target_name
            FROM relations r
            JOIN entities se ON se.entity_id = r.source_entity_id
            JOIN entities te ON te.entity_id = r.target_entity_id
            """
        ).fetchall()
        adjacency: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row["predicate"] not in allowed:
                continue
            edge = {
                "relation_id": row["relation_id"],
                "source_entity_id": row["source_entity_id"],
                "source_name": row["source_name"],
                "target_entity_id": row["target_entity_id"],
                "target_name": row["target_name"],
                "predicate": row["predicate"],
                "observation_ids": json.loads(row["observation_ids_json"]),
                "confidence": float(row["confidence"]),
                "extraction_method": json.loads(row["payload_json"]).get("extraction_method"),
            }
            adjacency.setdefault(edge["source_entity_id"], []).append(
                {**edge, "next_entity_id": edge["target_entity_id"], "traversal_direction": "outbound"}
            )
            adjacency.setdefault(edge["target_entity_id"], []).append(
                {**edge, "next_entity_id": edge["source_entity_id"], "traversal_direction": "inbound"}
            )

        entity_rows = self.connection.execute(
            "SELECT entity_id, canonical_name FROM entities"
        ).fetchall()
        entity_names = {row["entity_id"]: row["canonical_name"] for row in entity_rows}
        frontier = [
            {
                "seed_entity_id": entity_id,
                "current_entity_id": entity_id,
                "nodes": (entity_id,),
                "edges": (),
                "observation_ids": (),
                "score": 1.0,
            }
            for entity_id in dict.fromkeys(entity_ids)
            if entity_id in entity_names
        ]
        paths: list[dict[str, Any]] = []
        observation_ids: set[str] = set()
        for hop in range(1, max_hops + 1):
            candidates: list[dict[str, Any]] = []
            for state in frontier:
                neighbors = adjacency.get(state["current_entity_id"], [])
                degree_penalty = 1.0 / max(1.0, (len(neighbors) ** 0.5))
                for edge in neighbors:
                    next_entity_id = edge["next_entity_id"]
                    if next_entity_id in state["nodes"]:
                        continue
                    edge_score = edge["confidence"] * degree_penalty * (0.85 ** (hop - 1))
                    next_observation_ids = tuple(
                        sorted(set(state["observation_ids"]) | set(edge["observation_ids"]))
                    )
                    candidates.append(
                        {
                            "seed_entity_id": state["seed_entity_id"],
                            "current_entity_id": next_entity_id,
                            "nodes": (*state["nodes"], next_entity_id),
                            "edges": (*state["edges"], edge),
                            "observation_ids": next_observation_ids,
                            "score": state["score"] * edge_score,
                        }
                    )
            candidates.sort(key=lambda state: state["score"], reverse=True)
            next_frontier: list[dict[str, Any]] = []
            seen_states: set[tuple[str, str, tuple[str, ...]]] = set()
            for state in candidates:
                state_key = (state["seed_entity_id"], state["current_entity_id"], state["nodes"])
                if state_key in seen_states:
                    continue
                seen_states.add(state_key)
                next_frontier.append(state)
                observation_ids.update(state["observation_ids"])
                last_edge = state["edges"][-1]
                paths.append(
                    {
                        "path_id": stable_id(
                            "graph_path",
                            {
                                "seed_entity_id": state["seed_entity_id"],
                                "nodes": state["nodes"],
                                "edges": [
                                    {
                                        "relation_id": edge["relation_id"],
                                        "traversal_direction": edge["traversal_direction"],
                                    }
                                    for edge in state["edges"]
                                ],
                            },
                        ),
                        "hop": hop,
                        "seed_entity_id": state["seed_entity_id"],
                        "seed_name": entity_names[state["seed_entity_id"]],
                        "source_entity_id": last_edge["source_entity_id"],
                        "source_name": last_edge["source_name"],
                        "predicate": last_edge["predicate"],
                        "target_entity_id": last_edge["target_entity_id"],
                        "target_name": last_edge["target_name"],
                        "end_entity_id": state["current_entity_id"],
                        "end_name": entity_names[state["current_entity_id"]],
                        "observation_ids": list(state["observation_ids"]),
                        "confidence": state["score"],
                        "path_score": state["score"],
                        "extraction_method": last_edge["extraction_method"],
                        "path_edges": [
                            {
                                "relation_id": edge["relation_id"],
                                "source_entity_id": edge["source_entity_id"],
                                "source_name": edge["source_name"],
                                "predicate": edge["predicate"],
                                "target_entity_id": edge["target_entity_id"],
                                "target_name": edge["target_name"],
                                "observation_ids": edge["observation_ids"],
                                "confidence": edge["confidence"],
                                "extraction_method": edge["extraction_method"],
                                "traversal_direction": edge["traversal_direction"],
                            }
                            for edge in state["edges"]
                        ],
                    }
                )
                if len(next_frontier) >= beam_width:
                    break
            if not next_frontier:
                break
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
