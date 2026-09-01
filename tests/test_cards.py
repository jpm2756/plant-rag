from ingestion.cards import build_documents, filter_metadata, pivot, species_row
from ingestion.usda_client import strip_tags

CHARACTERISTICS = [
    {"PlantCharacteristicName": "Shade Tolerance", "PlantCharacteristicValue": "Tolerant"},
    {"PlantCharacteristicName": "Drought Tolerance", "PlantCharacteristicValue": "Low"},
    {"PlantCharacteristicName": "pH, Minimum", "PlantCharacteristicValue": "4.0"},
    {"PlantCharacteristicName": "pH, Maximum", "PlantCharacteristicValue": "6.5"},
    {"PlantCharacteristicName": "Height, Mature (feet)", "PlantCharacteristicValue": "60.0"},
    {"PlantCharacteristicName": "Toxicity", "PlantCharacteristicValue": "None"},
    {"PlantCharacteristicName": "Bloom Period", "PlantCharacteristicValue": "Spring"},
    # a cultivar-qualified duplicate must lose to the species-level value
    {
        "PlantCharacteristicName": "Shade Tolerance",
        "PlantCharacteristicValue": "Intolerant",
        "CultivarName": "Nana",
    },
    {"PlantCharacteristicName": "Growth Rate", "PlantCharacteristicValue": "None Given"},
]

STUB = {
    "id": 15309,
    "symbol": "ABBA",
    "scientificName": "<i>Abies balsamea</i> (L.) Mill.",
    "scientificNameWithoutAuthor": "Abies balsamea",
    "commonName": "balsam fir",
    "familyName": "Pinaceae",
}

PROFILE = {
    "Durations": ["Perennial"],
    "GrowthHabits": ["Tree"],
    "NativeStatuses": [{"Region": "L48", "Type": "Native"}, {"Region": "AK", "Type": "Introduced"}],
    "FactSheetUrls": ["/DocumentLibrary/factsheet/pdf/fs_abba.pdf"],
    "OtherCommonNames": ["balsam"],
    "Synonyms": [],
}


def test_strip_tags():
    assert strip_tags("<i>Abies balsamea</i> (L.) Mill.") == "Abies balsamea (L.) Mill."


def test_pivot_prefers_species_level_and_drops_placeholders():
    traits = pivot(CHARACTERISTICS)
    assert traits["Shade Tolerance"] == "Tolerant"
    assert "Growth Rate" not in traits


def test_species_row_types_numeric_and_categorical_fields():
    row = species_row(STUB, PROFILE, CHARACTERISTICS, wetland=[])
    assert row["id"] == 15309
    assert row["scientific_name"] == "Abies balsamea (L.) Mill."
    assert row["genus"] == "Abies"
    assert row["ph_min"] == 4.0 and row["ph_max"] == 6.5
    assert row["height_mature_ft"] == 60.0
    assert row["shade_tolerance"] == "Tolerant"
    assert row["growth_habit"] == ["Tree"]
    assert row["profile_url"].endswith("/plant-profile/ABBA")


def test_build_documents_emits_one_chunk_per_section_with_prose():
    row = species_row(STUB, PROFILE, CHARACTERISTICS, wetland=[])
    docs = build_documents(row, wildlife={})
    assert [d["section"] for d in docs] == ["summary", "growth", "site", "propagation", "uses"]
    summary = docs[0]
    assert summary["doc_id"] == "15309::summary"
    assert "balsam fir" in summary["text"]
    assert "native to the lower 48 states" in summary["text"]
    assert summary["metadata"]["symbol"] == "ABBA"
    assert len(summary["content_hash"]) == 16


def test_filter_metadata_exposes_filterable_traits():
    row = species_row(STUB, PROFILE, CHARACTERISTICS, wetland=[])
    metadata = filter_metadata(row)
    assert metadata["native_regions"] == ["L48"]
    assert metadata["height_mature_ft"] == 60.0
    assert metadata["shade_tolerance"] == "Tolerant"
