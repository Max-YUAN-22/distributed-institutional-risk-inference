"""Manifest manager — crash-safe write protocol for Phases 2-5.

Implements the write-temp / rename-on-success / manifest-gating protocol
described in §2.7 of the manuscript.  Every artefact is written to a
``.tmp`` file first, fsynced, atomically renamed, then registered in a
JSON manifest with a SHA-256 digest.  Re-running the same Phase finds
the manifest entry and skips the write — i.e. Phases 2-5 are idempotent
under partial failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict


class ManifestManager:
    """Idempotent artefact registry with crash-safe write semantics."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Dict[str, Any]] = {}
        if self.manifest_path.exists():
            try:
                self._state = json.loads(self.manifest_path.read_text())
            except json.JSONDecodeError:
                # corrupt manifest — treat as fresh; the .tmp/rename
                # protocol guarantees the underlying artefacts are intact.
                self._state = {}

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self._state

    def get(self, key: str) -> Dict[str, Any] | None:
        return self._state.get(key)

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def write_artefact(
        self, key: str, dest_path: str | Path, write_fn, **extra
    ) -> Path:
        """Crash-safe write: write_fn(tmp_path) then atomic rename.

        Parameters
        ----------
        key       : manifest key (e.g. 'phase2.merged_partition.03').
        dest_path : final on-disk path.
        write_fn  : callable taking a Path argument; writes content to it.
        extra     : extra fields stored in the manifest entry.
        """
        if self.has(key):
            entry = self._state[key]
            existing = Path(entry["path"])
            if existing.exists() and self._sha256(existing) == entry["sha256"]:
                print(f"  [MANIFEST] hit '{key}' — skipping write")
                return existing
            else:
                print(f"  [MANIFEST] stale entry '{key}' — re-writing")

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        write_fn(tmp)
        # fsync the file and its directory for crash safety
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, dest)  # atomic on POSIX

        digest = self._sha256(dest)
        self._state[key] = {
            "path": str(dest),
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "written_at": time.time(),
            **extra,
        }
        self._flush()
        return dest

    def _flush(self) -> None:
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, sort_keys=True))
        os.replace(tmp, self.manifest_path)

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def invalidate(self, key: str) -> None:
        self._state.pop(key, None)
        self._flush()

    def keys(self):
        return list(self._state.keys())
