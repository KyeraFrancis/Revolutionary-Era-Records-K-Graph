"""Text normalization shared by evaluation, entity extraction, and linking."""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s'&-]")
_TITLES = re.compile(
    r"\b(mr|mrs|miss|dr|rev|capt|captain|col|colonel|lieut|lt|gen|general|sergt|serjeant|"
    r"sgt|maj|major|esq|esquire|hon|jr|sr|private|priv|pvt)\.?\b",
    re.I,
)
# Common 18th-century spellings / abbreviations -> modern
_ARCHAIC = {
    "serjeant": "sergeant",
    "servt": "servant",
    "regt": "regiment",
    "comp": "company",
    "colo": "colonel",
    "feby": "february",
    "sepr": "september",
    "octr": "october",
    "novr": "november",
    "decr": "december",
    "ye": "the",
}


def normalize_for_cer(s: str | None) -> str:
    """Light normalization used before CER/WER so trivial differences don't dominate."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ſ", "s")  # long s
    s = re.sub(r"\[.*?\]", " ", s)  # editorial notes like [illegible], [signed]
    s = _WS.sub(" ", s).strip().lower()
    return s


def normalize_name(s: str | None) -> str:
    """Canonical key for a person name: lowercase, strip titles/punct, expand archaic forms."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).replace("ſ", "s")
    s = _TITLES.sub(" ", s)
    # "Perkins, Joseph" -> "Joseph Perkins" (only when exactly one comma splits two parts)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 2 and all(parts):
        s = f"{parts[1]} {parts[0]}"
    s = _PUNCT.sub(" ", s)
    tokens = [_ARCHAIC.get(t, t) for t in s.lower().split()]
    return " ".join(tokens)


def normalize_place(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip().lower()
