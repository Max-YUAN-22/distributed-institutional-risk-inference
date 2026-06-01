"""Base extractor interface — Phase 1 of the distributed pipeline.

Every source-system extractor implements BaseExtractor, exposing a stable
contract so that the downstream stream-merger (Phase 2) is independent of
any individual source schema.  Adding a new source system requires only a
new subclass; Phases 2-5 are unchanged.
"""
from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# canonical output column names (must be produced by every extractor)
REQUIRED_COLUMNS = frozenset({
    "student_id",  # institution-internal pseudonymised key
    "semester",    # integer 1-4 (enrolment-relative)
    "source",      # string: "ams" | "iucp" | "lms" | "sas"
})


class BaseExtractor(abc.ABC):
    """Abstract base for all Phase-1 source-system extractors.

    Subclasses must implement:
        - ``_raw_fetch()``: pulls records from the source system (SQL, API, file).
        - ``_transform(df)``: maps raw columns to the canonical schema.
        - ``source_name``: unique string identifier for this source.

    The public ``extract()`` method wraps these with timing, schema validation,
    and Parquet shard output.
    """

    source_name: str = ""

    def __init__(self, output_dir: str | Path, config: Dict[str, Any] | None = None):
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def _raw_fetch(self) -> pd.DataFrame:
        """Return raw records from the source system."""
        ...

    @abc.abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw columns → canonical schema columns."""
        ...

    def _validate(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{self.source_name} extractor produced DataFrame missing "
                f"required columns: {missing}"
            )
        if df["student_id"].isna().any():
            raise ValueError(f"{self.source_name}: null student_id in output")

    def extract(self, semester_range: tuple[int, int] = (1, 4)) -> Path:
        """Run the full extract-transform-write cycle.

        Parameters
        ----------
        semester_range : (first, last) inclusive semester integers.

        Returns
        -------
        Path to the written Parquet shard.
        """
        t0 = time.perf_counter()
        raw = self._raw_fetch()
        df = self._transform(raw)
        df = df[df["semester"].between(*semester_range)]
        self._validate(df)

        shard_path = self.output_dir / f"{self.source_name}.parquet"
        pq.write_table(pa.Table.from_pandas(df), str(shard_path))

        elapsed = time.perf_counter() - t0
        print(
            f"  [{self.source_name.upper()}] {len(df):,} rows → "
            f"{shard_path.name}  ({elapsed:.2f}s)"
        )
        return shard_path
