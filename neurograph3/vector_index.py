"""Optional persistent ANN indexes for the authoritative SQLite embeddings."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


class ANNUnavailable(RuntimeError):
    """Raised when no optional ANN implementation is installed."""


class ANNIndex:
    """A rebuildable HNSW index with an explicit SQLite/brute-force fallback.

    The index stores only derived vector data and an ID mapping. SQLite remains
    authoritative, so a corrupt or stale ANN file can be discarded and rebuilt.
    """

    def __init__(self, root: str | Path, backend: str = "auto", ef_search: int = 64):
        if backend not in {"auto", "hnswlib", "faiss"}:
            raise ValueError("backend must be auto, hnswlib, or faiss")
        if ef_search < 1:
            raise ValueError("ef_search must be positive")
        self.root = Path(root)
        self.backend = self._select_backend(backend)
        self.ef_search = ef_search
        self.metadata_path = self.root / "metadata.json"
        self.index_path = self.root / ("index.bin" if self.backend != "unavailable" else "index.unavailable")
        self._metadata: dict[str, Any] | None = None

    @staticmethod
    def _select_backend(requested: str) -> str:
        if requested in {"hnswlib", "auto"}:
            try:
                import hnswlib  # noqa: F401

                return "hnswlib"
            except ImportError:
                if requested == "hnswlib":
                    raise ANNUnavailable("hnswlib is not installed")
        if requested in {"faiss", "auto"}:
            try:
                import faiss  # noqa: F401

                return "faiss"
            except ImportError:
                if requested == "faiss":
                    raise ANNUnavailable("faiss is not installed")
        return "unavailable"

    @property
    def available(self) -> bool:
        return self.backend != "unavailable"

    @property
    def backend_name(self) -> str:
        return self.backend

    def rebuild(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.available:
            raise ANNUnavailable("no ANN backend is installed; install hnswlib or faiss")
        if not records:
            raise ValueError("cannot build an ANN index with no embedding records")
        dimensions = int(records[0]["dimensions"])
        model = records[0]["model"]
        if dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        for record in records:
            if int(record["dimensions"]) != dimensions or len(record["vector"]) != dimensions:
                raise ValueError("all embedding records must have the same dimensions")
            if record["model"] != model:
                raise ValueError("all embedding records must use the same embedding model")

        vectors = self._normalized_vectors(records, dimensions)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / f"index-{uuid.uuid4().hex}.bin"
        if self.backend == "hnswlib":
            self._rebuild_hnsw(vectors, dimensions)
        else:
            self._rebuild_faiss(vectors, dimensions)
        metadata = {
            "backend": self.backend,
            "dimensions": dimensions,
            "model": model,
            "observation_ids": [record["observation_id"] for record in records],
            "count": len(records),
            "metric": "cosine",
            "ef_search": self.ef_search,
            "index_file": self.index_path.name,
        }
        self._atomic_write_json(self.metadata_path, metadata)
        self._metadata = metadata
        return metadata

    @staticmethod
    def _normalized_vectors(records: list[dict[str, Any]], dimensions: int) -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional backend environments
            raise ANNUnavailable("NumPy is required by the ANN backend") from exc
        vectors = np.asarray([record["vector"] for record in records], dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        if vectors.shape != (len(records), dimensions):
            raise ValueError("embedding vectors do not match their declared dimensions")
        return vectors

    def _rebuild_hnsw(self, vectors: Any, dimensions: int) -> None:
        import hnswlib

        index = hnswlib.Index(space="cosine", dim=dimensions)
        index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
        index.add_items(vectors, list(range(len(vectors))))
        index.set_ef(max(self.ef_search, 16))
        self._atomic_write_index(lambda path: index.save_index(str(path)))

    def _rebuild_faiss(self, vectors: Any, dimensions: int) -> None:
        import faiss

        index = faiss.IndexHNSWFlat(dimensions, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = max(self.ef_search, 16)
        index.add(vectors)
        self._atomic_write_index(lambda path: faiss.write_index(index, str(path)))

    def _atomic_write_index(self, writer: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="index.", suffix=".tmp", dir=self.root)
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            writer(temporary_path)
            os.replace(temporary_path, self.index_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix="metadata.", suffix=".tmp", dir=path.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            try:
                self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise ANNUnavailable(f"ANN index metadata is unavailable: {exc}") from exc
        if self._metadata.get("backend") != self.backend:
            raise ANNUnavailable("ANN index backend does not match the configured backend")
        self.index_path = self.root / self._metadata.get("index_file", "index.bin")
        if not self.index_path.exists():
            raise ANNUnavailable("ANN index file is unavailable; rebuild the index")
        return self._metadata

    def search(self, query_vector: list[float], limit: int = 20) -> list[dict[str, float | str]]:
        if not self.available:
            raise ANNUnavailable("no ANN backend is installed; using SQLite fallback")
        if limit < 1:
            return []
        metadata = self._load_metadata()
        dimensions = int(metadata["dimensions"])
        if len(query_vector) != dimensions:
            raise ValueError(f"query vector dimensions differ from index: {len(query_vector)} != {dimensions}")
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional backend environments
            raise ANNUnavailable("NumPy is required by the ANN backend") from exc
        query = np.asarray([query_vector], dtype="float32")
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        if float(norm[0][0]) == 0:
            return []
        query = query / norm
        if self.backend == "hnswlib":
            import hnswlib

            index = hnswlib.Index(space="cosine", dim=dimensions)
            index.load_index(str(self.index_path), max_elements=int(metadata["count"]))
            index.set_ef(max(self.ef_search, 16))
            labels, distances = index.knn_query(query, k=min(limit, int(metadata["count"])))
            scores = [1.0 - float(distance) for distance in distances[0]]
        else:
            import faiss

            index = faiss.read_index(str(self.index_path))
            index.hnsw.efSearch = max(self.ef_search, 16)
            scores_array, labels_array = index.search(query, min(limit, int(metadata["count"])))
            labels = labels_array
            scores = [float(score) for score in scores_array[0]]
        observation_ids = metadata["observation_ids"]
        return [
            {"observation_id": observation_ids[int(label)], "vector_score": score}
            for label, score in zip(labels[0], scores)
            if int(label) >= 0
        ]
