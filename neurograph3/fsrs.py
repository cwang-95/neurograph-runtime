"""Small adapter around the existing ZenBrain FSRS implementation.

The OpenClaw runtime owns the JavaScript package.  Graph 3.0 only owns the
boundary: JSON in, JSON out, and explicit feedback-driven state changes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


_NODE_PROGRAM = r"""
const z = require('@zensation/algorithms');
const [command, payload, value, afterDays] = process.argv.slice(1);

function parseMemory() {
  const memory = JSON.parse(payload || '{}');
  const fsrs = memory.schedulers && memory.schedulers.fsrs;
  if (!fsrs) throw new Error('no fsrs scheduler');
  if (typeof fsrs.nextReview === 'string') fsrs.nextReview = new Date(fsrs.nextReview);
  return memory;
}

function main() {
  if (command === 'new') {
    return { schedulers: { fsrs: z.initFromDecayClass('normal_decay') }, createdAt: Date.now() };
  }
  if (command === 'decay') {
    const memory = parseMemory();
    const days = Number(value || 7);
    const later = new Date(Date.now() + days * 24 * 3600 * 1000);
    const r = z.getRetrievability(memory.schedulers.fsrs, later);
    return { days, retrievability: r, retrievabilityPct: (r * 100).toFixed(1) };
  }
  if (command === 'recall') {
    const memory = parseMemory();
    const quality = Number(value || 4);
    const now = new Date(Date.now() + Number(afterDays || 7) * 24 * 3600 * 1000);
    const r = z.getRetrievability(memory.schedulers.fsrs, now);
    memory.schedulers.fsrs = z.updateAfterRecall(memory.schedulers.fsrs, quality, r, now);
    return { memory, retrievability: r };
  }
  throw new Error(`unknown command: ${command}`);
}

process.stdout.write(JSON.stringify(main()));
"""


class FSRSAdapterError(RuntimeError):
    """Raised when the external ZenBrain runtime cannot serve a request."""


class NodeZenBrainFSRS:
    """Use the existing Node FSRS package without importing it into Python."""

    def __init__(
        self,
        *,
        node_path: str | Path | None = None,
        node_modules: str | Path | None = None,
        timeout: float = 15.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ):
        self.node_path = str(node_path or shutil.which("node") or "/usr/local/bin/node")
        default_modules = Path.home() / ".openclaw" / "workspace" / "memory_rag" / "server" / "node_modules"
        self.node_modules = str(node_modules or os.environ.get("NEUROGRAPH_ZENBRAIN_NODE_MODULES", default_modules))
        self.timeout = timeout
        self._runner = runner or subprocess.run

    def _run(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
        *values: int | float,
    ) -> dict[str, Any]:
        args = [self.node_path, "-e", _NODE_PROGRAM, command]
        if command != "new":
            args.append(json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":")))
        args.extend(str(value) for value in values)
        env = os.environ.copy()
        env["NODE_PATH"] = self.node_modules
        try:
            completed = self._runner(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise FSRSAdapterError(f"FSRS subprocess failed: {exc}") from exc
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise FSRSAdapterError(f"FSRS subprocess exited {completed.returncode}: {detail}")
        try:
            result = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FSRSAdapterError("FSRS subprocess returned invalid JSON") from exc
        if isinstance(result, dict) and result.get("error"):
            raise FSRSAdapterError(str(result["error"]))
        return result

    def new_memory(self) -> dict[str, Any]:
        return self._run("new")

    def decay(self, memory: dict[str, Any], days: float = 7.0) -> float:
        result = self._run("decay", memory, days)
        return max(0.0, min(1.0, float(result["retrievability"])))

    def recall(self, memory: dict[str, Any], quality: int = 4, after_days: float = 7.0) -> dict[str, Any]:
        result = self._run("recall", memory, quality, after_days)
        updated = result.get("memory")
        if not isinstance(updated, dict):
            raise FSRSAdapterError("FSRS recall did not return updated memory")
        return updated


__all__ = ["FSRSAdapterError", "NodeZenBrainFSRS"]
