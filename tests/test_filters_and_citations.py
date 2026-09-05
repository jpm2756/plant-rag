from app.flow import _cited_indices
from app.llm import sanitize_filters


def test_sanitize_filters_drops_empty_unknown_and_out_of_vocabulary_values():
    assert sanitize_filters({"height_mature_ft": {}, "shade_tolerance": None}) == {}
    assert sanitize_filters({"nonexistent_field": "Shrub"}) == {}
    assert sanitize_filters({"shade_tolerance": ["Tolerant", "Very shady"]}) == {
        "shade_tolerance": ["Tolerant"]
    }
    assert sanitize_filters("not a dict") == {}


def test_sanitize_filters_normalises_casing_and_bare_numbers():
    assert sanitize_filters({"growth_habit": "shrub"}) == {"growth_habit": ["Shrub"]}
    assert sanitize_filters({"height_mature_ft": 6}) == {"height_mature_ft": {"lte": 6.0}}
    assert sanitize_filters({"height_mature_ft": {"lte": 6, "junk": 3}}) == {
        "height_mature_ft": {"lte": 6.0}
    }


def test_hardiness_zone_becomes_a_minimum_temperature_bound():
    assert sanitize_filters({"hardiness_zone": 5}) == {"temp_min_f": {"lte": -20.0}}
    assert sanitize_filters({"hardiness_zone": "6b"}) == {"temp_min_f": {"lte": -10.0}}
    assert sanitize_filters({"hardiness_zone": "not a zone"}) == {}


def test_soil_ph_becomes_a_containment_test_on_both_endpoints():
    assert sanitize_filters({"soil_ph": 6.5}) == {
        "ph_min": {"lte": 6.5},
        "ph_max": {"gte": 6.5},
    }


def test_cited_indices_parses_every_citation_style():
    assert _cited_indices("hardy [1] and shade tolerant [2][3]") == [1, 2, 3]
    assert _cited_indices("both apply [1, 2]") == [1, 2]
    assert _cited_indices("no citations here") == []
