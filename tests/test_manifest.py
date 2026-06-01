"""Manifest-manager smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.fusion import ManifestManager


def test_atomic_write_and_replay(tmp_path: Path):
    mgr = ManifestManager(tmp_path / "manifest.json")
    target = tmp_path / "X.npy"

    mgr.write_artefact("X", target, lambda p: np.save(p, np.zeros(8)))
    assert target.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "X" in manifest

    # second call should be idempotent and not raise
    mgr.write_artefact("X", target, lambda p: np.save(p, np.zeros(8)))


def test_corrupt_target_triggers_rewrite(tmp_path: Path):
    mgr = ManifestManager(tmp_path / "manifest.json")
    target = tmp_path / "Y.npy"
    mgr.write_artefact("Y", target, lambda p: np.save(p, np.ones(4)))

    # corrupt the file: SHA should mismatch
    target.write_bytes(b"corrupted")
    mgr.write_artefact("Y", target, lambda p: np.save(p, np.ones(4)))
    arr = np.load(target)
    assert (arr == 1).all()
