"""OpenAI-compatible embedding client for the local Qwen embedding service."""

from __future__ import annotations

import json
import urllib.request
from typing import Sequence


class EmbeddingError(RuntimeError):
    pass


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/v1/embeddings",
        model: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        timeout: float = 30.0,
        batch_size: int = 16,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > self.batch_size:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self.batch_size):
                vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
            return vectors

        return self._embed_batch(texts)

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": list(texts)}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except Exception as exc:  # pragma: no cover - exercised by service integration
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        try:
            rows = sorted(body["data"], key=lambda item: item["index"])
            vectors = [list(map(float, row["embedding"])) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("embedding response is not OpenAI-compatible") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(f"embedding count mismatch: requested {len(texts)}, got {len(vectors)}")
        return vectors
