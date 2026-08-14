"""Stable, content-addressed identifiers for Graph 3.0 entities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for IDs and fingerprints."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_hex(value: bytes | str) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def content_hash(value: bytes | str) -> str:
    """Return a full SHA-256 hash for immutable source content."""
    return sha256_hex(value)


def stable_id(namespace: str, value: Mapping[str, Any] | str) -> str:
    """Create a deterministic ID scoped by an entity namespace."""
    payload = value if isinstance(value, str) else canonical_json(value)
    digest = sha256_hex(f"{namespace}:\n{payload}")[:32]
    return f"{namespace}_{digest}"
