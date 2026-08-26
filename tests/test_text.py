from rcgraph.text import normalize_for_cer, normalize_name, normalize_place


def test_normalize_name_strips_titles_and_reorders_lastname_first():
    assert normalize_name("Perkins, Joseph") == "joseph perkins"
    assert normalize_name("Serjeant Perkins") == "perkins"
    assert normalize_name("Col. Charles Johnston") == "charles johnston"


def test_normalize_name_handles_long_s_and_unicode():
    assert normalize_name("Joſeph Perkins") == "joseph perkins"


def test_normalize_for_cer_drops_editorial_brackets():
    assert normalize_for_cer("Haverhill [signed] Charles  Johnston") == "haverhill charles johnston"


def test_normalize_place():
    assert normalize_place("Fairfield, Conn.") == "fairfield conn"
