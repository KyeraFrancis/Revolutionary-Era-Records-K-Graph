"""Loaders for the three Revolution Crossroads collections on Hugging Face.

All three are CC0. We pull only the columns we need and cache locally as parquet.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def _load_hf(dataset: str, revision: str | None, columns: list[str] | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train", revision=revision)
    if columns:
        keep = [c for c in columns if c in ds.column_names]
        ds = ds.select_columns(keep)
    return ds.to_pandas()


def _cached(cache_dir: Path, name: str, builder) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    return df


def load_nara(cfg: dict, with_truth_only: bool = True) -> pd.DataFrame:
    c = cfg["data"]["nara"]
    cols = [c["id_col"], c["file_col"], c["image_col"], c["truth_col"], c["names_col"],
            "title", "pageNumber", "pageImageType", "levelOfDescription", *c["baseline_cols"]]

    def build():
        df = _load_hf(c["dataset"], c["revision"], cols)
        if with_truth_only:
            df = df[df[c["truth_col"]].fillna("").str.strip().str.len() > 20]
        if c.get("sample_size"):
            df = df.sample(n=min(c["sample_size"], len(df)), random_state=42)
        return df.reset_index(drop=True)

    return _cached(Path(cfg["data"]["cache_dir"]), "nara", build)


def load_smithsonian(cfg: dict) -> pd.DataFrame:
    c = cfg["data"]["smithsonian"]
    cols = [c["id_col"], "title", "unitCode", "collectionsURL", c["text_col"],
            c["names_col"], c["places_col"], c["dates_col"], "indexed_topics"]
    return _cached(Path(cfg["data"]["cache_dir"]), "smithsonian",
                   lambda: _load_hf(c["dataset"], c["revision"], cols))


def load_chronam(cfg: dict) -> pd.DataFrame:
    c = cfg["data"]["chronam"]

    def build():
        df = _load_hf(c["dataset"], c["revision"])
        text_col = c["text_col"] if c["text_col"] in df.columns else next(
            (col for col in df.columns if "ocr" in col.lower()), None)
        id_col = c["id_col"] or next(
            (col for col in df.columns if "id" in col.lower()), df.columns[0])
        date_col = c["date_col"] or next((col for col in df.columns if "date" in col.lower()), None)
        out = pd.DataFrame({
            "id": df[id_col].astype(str),
            "text": df[text_col].fillna("") if text_col else "",
            "date": df[date_col] if date_col else None,
        })
        for extra in ("title", "newspaper", "city", "state", "page"):
            if extra in df.columns:
                out[extra] = df[extra]
        return out

    return _cached(Path(cfg["data"]["cache_dir"]), "chronam", build)


def download_images(df: pd.DataFrame, url_col: str, images_dir: str | Path,
                    timeout: int = 30) -> pd.Series:
    """Download page images once; return a Series of local paths aligned to df.index."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    session = requests.Session()
    for url in tqdm(df[url_col], desc="images"):
        name = hashlib.sha1(url.encode()).hexdigest()[:16] + Path(url).suffix.lower()
        dest = images_dir / name
        if not dest.exists():
            try:
                r = session.get(url, timeout=timeout)
                r.raise_for_status()
                dest.write_bytes(r.content)
            except (requests.RequestException, OSError):
                paths.append(None)
                continue
        paths.append(str(dest))
    return pd.Series(paths, index=df.index, name="image_path")
