"""Entity extraction from OCR text.

Two tiers:
  1. `extract_rules`  - regex for dates/years and a pass over pre-indexed names supplied by the
     source dataset (NARA `extractedNames`, Smithsonian `indexed_names`). Zero dependencies.
  2. `extract_spacy`  - spaCy NER for PERSON / GPE / ORG / DATE on free text (newspapers).

An LLM tier (Anthropic/OpenAI tool-calling for relation extraction) is a clean drop-in here
and is the natural "agentic" extension: the model decides which pages need a second look.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rcgraph.text import normalize_name, normalize_place

_YEAR = re.compile(r"\b(1[67]\d\d|18[0-4]\d)\b")
_NAME_LIST = re.compile(r"(?:names? found on this page include|additional names)\s*:?\s*(.+)", re.I)


@dataclass
class Entities:
    people: set[str] = field(default_factory=set)
    places: set[str] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    orgs: set[str] = field(default_factory=set)


def parse_name_field(raw: str | list | None) -> list[str]:
    """Handle both the NARA prose field and Smithsonian list field."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)) or hasattr(raw, "tolist"):
        items = list(raw)
    else:
        m = _NAME_LIST.search(str(raw))
        body = m.group(1) if m else str(raw)
        items = re.split(r"[;,]\s*|\s{2,}", body)
    out = []
    for it in items:
        it = str(it).strip().rstrip(".")
        if "[blank]" in it.lower() or len(it) < 3:
            continue
        out.append(it)
    return out


def extract_rules(text: str | None, names_field=None, places_field=None,
                  min_name_tokens: int = 2) -> Entities:
    e = Entities()
    for n in parse_name_field(names_field):
        key = normalize_name(n)
        if len(key.split()) >= min_name_tokens:
            e.people.add(key)
    if places_field is not None:
        for p in parse_name_field(places_field):
            e.places.add(normalize_place(p))
    if text:
        e.years.update(int(y) for y in _YEAR.findall(text))
    return e


def extract_spacy(text: str | None, nlp=None, min_name_tokens: int = 2) -> Entities:
    if not text:
        return Entities()
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    e = extract_rules(text)
    for ent in nlp(text[:100_000]).ents:
        if ent.label_ == "PERSON":
            key = normalize_name(ent.text)
            if len(key.split()) >= min_name_tokens:
                e.people.add(key)
        elif ent.label_ in ("GPE", "LOC"):
            e.places.add(normalize_place(ent.text))
        elif ent.label_ == "ORG":
            e.orgs.add(ent.text.strip())
    return e
