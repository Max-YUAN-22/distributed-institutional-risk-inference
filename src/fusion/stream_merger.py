"""Stream merger — Phase 2 of the distributed pipeline.

Merges per-source Parquet shards on the (student_id, semester) join key,
preserving source-attribution metadata.  The merge is partition-parallel:
the student-ID space is hash-partitioned into `n_partitions` shards, each
of which is merged independently.

Communication cost (per Section 2.3 of the manuscript):
    O(N * T * F / P)    one-time shuffle over P workers.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pyarrow.parquet as pq


JOIN_KEYS = ("student_id", "semester")


def _hash_partition(student_id, n_partitions: int) -> int:
    """Deterministic hash partition assignment."""
    # cheap, stable hash — int-safe for both string and integer IDs
    return abs(hash(str(student_id))) % n_partitions


class StreamMerger:
    """Partition-parallel outer join over per-source shards."""

    def __init__(self, n_partitions: int = 4, how: str = "outer"):
        self.n_partitions = n_partitions
        self.how = how

    def _load_shard(self, path: Path) -> pd.DataFrame:
        return pq.read_table(str(path)).to_pandas()

    def _merge_partition(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge a list of source-aligned DataFrames on (student_id, semester)."""
        if not dfs:
            return pd.DataFrame()
        merged = dfs[0]
        for nxt in dfs[1:]:
            # drop the 'source' column from non-leading shards to avoid suffix clash
            drop_cols = [c for c in ("source",) if c in nxt.columns]
            merged = merged.merge(
                nxt.drop(columns=drop_cols),
                on=list(JOIN_KEYS),
                how=self.how,
                suffixes=("", "_dup"),
            )
        # drop duplicate columns introduced by suffix collision
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)
        return merged

    def merge(self, shard_paths: Iterable[Path]) -> pd.DataFrame:
        """Read each shard, hash-partition by student_id, merge, concatenate."""
        t0 = time.perf_counter()
        shards = [self._load_shard(Path(p)) for p in shard_paths]
        if not shards:
            raise ValueError("StreamMerger.merge() received no shards")

        # assign partition id and split per source
        partitioned = {p: [] for p in range(self.n_partitions)}
        for shard in shards:
            shard = shard.copy()
            shard["_pid"] = shard["student_id"].apply(
                lambda s: _hash_partition(s, self.n_partitions)
            )
            for pid, grp in shard.groupby("_pid"):
                partitioned[pid].append(grp.drop(columns="_pid"))

        # merge each partition independently
        merged_parts = [
            self._merge_partition(parts) for parts in partitioned.values()
        ]
        merged = pd.concat(
            [m for m in merged_parts if not m.empty], ignore_index=True
        )
        merged = merged.sort_values(list(JOIN_KEYS)).reset_index(drop=True)

        elapsed = time.perf_counter() - t0
        print(
            f"  [MERGE] {len(shards)} shards × {len(merged):,} rows merged "
            f"on {self.n_partitions} partitions  ({elapsed:.2f}s)"
        )
        return merged
