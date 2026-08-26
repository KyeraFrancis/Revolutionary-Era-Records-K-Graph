from rcgraph.entities import extract_rules, parse_name_field
from rcgraph.link import Mention, link_mentions, years_compatible


def test_parse_nara_prose_names():
    raw = "Names found on this page include: Joseph Perkins; Andrew Henderson; [BLANK] Seabrook."
    assert parse_name_field(raw) == ["Joseph Perkins", "Andrew Henderson"]


def test_parse_smithsonian_list_names():
    names = ["Green, Frederick", "United States Mint"]
    assert parse_name_field(names) == names


def test_extract_rules_years_and_people():
    e = extract_rules("Haverhill July 27, 1782 ... Act: June 7th 1832",
                      "Names found on this page include: Charles Johnston.")
    assert e.years == {1782, 1832}
    assert e.people == {"charles johnston"}


def test_years_compatible_window():
    assert years_compatible(frozenset({1782}), frozenset({1832}))
    assert not years_compatible(frozenset({1700}), frozenset({1832}))
    assert years_compatible(frozenset(), frozenset({1832}))


def test_link_mentions_cross_collection_only():
    ms = [
        Mention("nara", "n1", "joseph perkins", frozenset({1782})),
        Mention("nara", "n2", "joseph perkins", frozenset({1782})),
        Mention("chronam", "c1", "jos perkins", frozenset({1790})),
        Mention("smithsonian", "s1", "joseph perkins", frozenset()),
        Mention("smithsonian", "s2", "frederick green", frozenset()),
    ]
    links = link_mentions(ms, threshold=80)
    pairs = {(lk.a.record_id, lk.b.record_id) for lk in links}
    assert ("n1", "n2") not in pairs          # same collection skipped
    assert ("n1", "s1") in pairs               # exact cross-collection
    assert all(lk.score >= 80 for lk in links)
    assert not any("green" in lk.a.name or "green" in lk.b.name for lk in links)
