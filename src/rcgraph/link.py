"""Cross-collection linking: match person names across NARA, Smithsonian, and newspapers.

Strategy (v1): blocking on surname, then rapidfuzz token_set_ratio over normalized names,
with a year-overlap sanity check when both sides have years. Every link carries a score so
the graph can filter by confidence and a reviewer can audit low-confidence edges.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rapidfuzz import fuzz


@dataclass(frozen=True)
class Mention:
    collection: str      # "nara" | "smithsonian" | "chronam"
    record_id: str
    name: str            # normalized
    years: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Link:
    a: Mention
    b: Mention
    score: float
    reason: str


def _surname(name: str) -> str:
    return name.split()[-1] if name else ""


def years_compatible(a: frozenset[int], b: frozenset[int], window: int = 60) -> bool:
    """Two mentions can be the same person if any year pair falls within `window` years."""
    if not a or not b:
        return True
    return any(abs(x - y) <= window for x in a for y in b)


def link_mentions(mentions: list[Mention], threshold: int = 92,
                  cross_collection_only: bool = True) -> list[Link]:
    blocks: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        if m.name:
            blocks[_surname(m.name)].append(m)
    links: list[Link] = []
    for group in blocks.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if cross_collection_only and a.collection == b.collection:
                    continue
                score = fuzz.token_set_ratio(a.name, b.name)
                if score < threshold:
                    continue
                if not years_compatible(a.years, b.years):
                    continue
                reason = "exact" if a.name == b.name else f"fuzzy:{score:.0f}"
                links.append(Link(a, b, float(score), reason))
    return links
