import math

import pandas as pd

from rcgraph.metrics import cer, sliced_scores, wer


def test_cer_and_wer_perfect_match():
    assert cer(["abc"], ["abc"]) == 0.0
    assert wer(["a b c"], ["a b c"]) == 0.0


def test_cer_single_substitution():
    assert math.isclose(cer(["abcd"], ["abxd"]), 0.25)


def test_empty_refs_are_skipped():
    assert math.isnan(cer([""], ["x"]))


def test_sliced_scores_has_all_and_per_slice_rows():
    df = pd.DataFrame({
        "ref": ["abcd", "abcd", "wxyz"],
        "sys": ["abcd", "abxd", "wxyz"],
        "kind": ["form", "letter", "letter"],
    })
    out = sliced_scores(df, "ref", ["sys"], ["kind"])
    assert set(out["slice"]) == {"all", "kind"}
    all_row = out[(out.slice == "all")].iloc[0]
    assert math.isclose(all_row.cer, 1 / 12)
    letter = out[(out.slice == "kind") & (out.value == "letter")].iloc[0]
    assert math.isclose(letter.cer, 1 / 8)
