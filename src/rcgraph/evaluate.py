"""Benchmark the fine-tuned model against the two OCR baselines shipped in the dataset.

Baselines:
  fs_extractedText  - NARA's existing full-text extraction
  RevX_OCR_text     - Revolution Crossroads' VLM OCR (Chandra OCR 2)
  trocr_finetuned   - ours
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rcgraph.metrics import sliced_scores
from rcgraph.text import normalize_for_cer


def evaluate(cfg: dict, model_dir: str | Path, split_parquet: str | Path | None = None,
             limit: int | None = None) -> pd.DataFrame:
    from rcgraph.ocr.predict import OCRModel

    c = cfg["data"]["nara"]
    o = cfg["ocr"]
    split_parquet = split_parquet or Path(o["output_dir"]) / "split.parquet"
    df = pd.read_parquet(split_parquet)
    df = df[df.split == "eval"]
    if limit:
        df = df.head(limit)

    model = OCRModel(model_dir)
    df["trocr_finetuned"] = [model.transcribe(p) for p in tqdm(df.image_path, desc="ocr")]

    systems = [*c["baseline_cols"], "trocr_finetuned"]
    norm = df.copy()
    norm[c["truth_col"]] = norm[c["truth_col"]].map(normalize_for_cer)
    for s in systems:
        norm[s] = norm[s].map(normalize_for_cer)

    report = sliced_scores(norm, c["truth_col"], systems, cfg["eval"]["slices"])
    out = Path(cfg["eval"]["report_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(orient="records"), indent=2))
    df.to_parquet(out.with_suffix(".predictions.parquet"), index=False)
    return report
