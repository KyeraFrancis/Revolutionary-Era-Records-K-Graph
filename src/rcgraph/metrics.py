"""CER / WER with per-slice reporting. Pure functions so they are trivially testable."""
from __future__ import annotations

from collections.abc import Sequence

import jiwer
import pandas as pd


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    if not pairs:
        return float("nan")
    r, h = zip(*pairs, strict=True)
    return float(jiwer.cer(list(r), list(h)))


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    if not pairs:
        return float("nan")
    r, h = zip(*pairs, strict=True)
    return float(jiwer.wer(list(r), list(h)))


def sliced_scores(df: pd.DataFrame, ref_col: str, hyp_cols: list[str],
                  slice_cols: list[str] | None = None) -> pd.DataFrame:
    """Return a tidy table: one row per (slice, value, system) with cer, wer, n."""
    rows = []
    groups = [("all", "all", df)]
    for sc in slice_cols or []:
        if sc in df.columns:
            groups += [(sc, str(v), g) for v, g in df.groupby(sc, dropna=False)]
    for slice_name, value, g in groups:
        for hyp in hyp_cols:
            rows.append({
                "slice": slice_name, "value": value, "system": hyp, "n": len(g),
                "cer": cer(g[ref_col].tolist(), g[hyp].fillna("").tolist()),
                "wer": wer(g[ref_col].tolist(), g[hyp].fillna("").tolist()),
            })
    return pd.DataFrame(rows)
