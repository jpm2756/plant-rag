"""Turn USDA characteristic triples into retrievable prose documents.

Structured ``{name, value}`` triples retrieve badly; prose retrieves well. Each
species becomes one summary card plus four section chunks, with the raw values
kept in chunk metadata so they can be used as hard filters at query time.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ingestion.usda_client import PROFILE_URL, strip_tags

SECTIONS = ("summary", "growth", "site", "propagation", "uses")

# characteristic name -> (column, caster)
NUMERIC_FIELDS: dict[str, str] = {
    "pH, Minimum": "ph_min",
    "pH, Maximum": "ph_max",
    "Height, Mature (feet)": "height_mature_ft",
    "Temperature, Minimum (°F)": "temp_min_f",
    "Precipitation, Minimum": "precip_min_in",
    "Precipitation, Maximum": "precip_max_in",
    "Frost Free Days, Minimum": "frost_free_days_min",
    "Root Depth, Minimum (inches)": "root_depth_min_in",
}

TEXT_FIELDS: dict[str, str] = {
    "Drought Tolerance": "drought_tolerance",
    "Shade Tolerance": "shade_tolerance",
    "Salinity Tolerance": "salinity_tolerance",
    "Fire Tolerance": "fire_tolerance",
    "Moisture Use": "moisture_use",
    "Growth Rate": "growth_rate",
    "Growth Form": "growth_form",
    "Lifespan": "lifespan",
    "Bloom Period": "bloom_period",
    "Flower Color": "flower_color",
    "Foliage Color": "foliage_color",
    "Fruit/Seed Color": "fruit_color",
    "Toxicity": "toxicity",
    "Nitrogen Fixation": "nitrogen_fixation",
    "Palatable Human": "palatable_human",
    "Fertility Requirement": "fertility_requirement",
    "Active Growth Period": "active_growth_period",
    "Leaf Retention": "leaf_retention",
    "Adapted to Coarse Textured Soils": "coarse_soil",
    "Adapted to Medium Textured Soils": "medium_soil",
    "Adapted to Fine Textured Soils": "fine_soil",
}

PROPAGATION_KEYS = (
    "Propagated by Seed",
    "Propagated by Cuttings",
    "Propagated by Bare Root",
    "Propagated by Container",
    "Propagated by Bulb",
    "Propagated by Corm",
    "Propagated by Sod",
    "Propagated by Sprigs",
    "Propagated by Tubers",
)

PRODUCT_KEYS = (
    "Berry/Nut/Seed Product",
    "Christmas Tree Product",
    "Fodder Product",
    "Lumber Product",
    "Naval Store Product",
    "Nursery Stock Product",
    "Post Product",
    "Pulpwood Product",
    "Veneer Product",
)

WETLAND_CODES = {
    "OBL": "obligate wetland",
    "FACW": "facultative wetland",
    "FAC": "facultative",
    "FACU": "facultative upland",
    "UPL": "upland",
    "NL": "not listed",
}


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def pivot(characteristics: list[dict[str, Any]]) -> dict[str, str]:
    """Collapse triples into ``name -> value``.

    A species can carry duplicate names for cultivars/synonyms; the row without a
    cultivar or synonym qualifier is the species-level value and wins.
    """
    best: dict[str, tuple[int, str]] = {}
    for row in characteristics:
        name = row.get("PlantCharacteristicName")
        value = row.get("PlantCharacteristicValue")
        if not name or value in (None, "", "None Given"):
            continue
        rank = 0 if not row.get("CultivarName") and not row.get("SynonymName") else 1
        current = best.get(name)
        if current is None or rank < current[0]:
            best[name] = (rank, str(value).strip())
    return {name: value for name, (_, value) in best.items()}


def _yes(traits: dict[str, str], key: str) -> bool:
    return traits.get(key, "").strip().lower() in {"yes", "true"}


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def species_row(
    stub: dict[str, Any],
    profile: dict[str, Any],
    characteristics: list[dict[str, Any]],
    wetland: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the flat ``species`` table row (typed filter columns + raw JSON)."""
    traits = pivot(characteristics)
    scientific = strip_tags(stub.get("scientificName")) or strip_tags(profile.get("ScientificName"))
    no_author = stub.get("scientificNameWithoutAuthor") or strip_tags(
        profile.get("ScientificNameWithoutAuthor")
    )
    symbol = stub.get("symbol") or profile.get("Symbol")

    wetland_designations = []
    for entry in wetland or []:
        for designation in entry.get("WetlandDesignations") or []:
            region = designation.get("Region")
            code = designation.get("WetlandCode")
            if region and code:
                wetland_designations.append(f"{region}: {code}")

    row: dict[str, Any] = {
        "id": int(stub["id"]),
        "symbol": symbol,
        "scientific_name": scientific,
        "scientific_no_author": no_author,
        "common_name": stub.get("commonName") or profile.get("CommonName"),
        "family": stub.get("familyName") or profile.get("Family"),
        "genus": (no_author or "").split(" ")[0] or None,
        "group_name": profile.get("Group"),
        "duration": profile.get("Durations") or [],
        "growth_habit": profile.get("GrowthHabits") or [],
        "native_status": profile.get("NativeStatuses") or [],
        "states": [],
        "other_common_names": profile.get("OtherCommonNames") or [],
        "synonyms": [strip_tags(s.get("ScientificName")) for s in (profile.get("Synonyms") or [])][
            :20
        ],
        "wetland_status": "; ".join(wetland_designations) or None,
        "factsheet_urls": profile.get("FactSheetUrls") or [],
        "plantguide_urls": profile.get("PlantGuideUrls") or [],
        "profile_url": PROFILE_URL.format(symbol=symbol) if symbol else None,
        "n_characteristics": len(traits),
        "raw_characteristics": traits,
    }
    for name, column in NUMERIC_FIELDS.items():
        row[column] = _to_float(traits.get(name))
    for name, column in TEXT_FIELDS.items():
        row[column] = traits.get(name)
    return row


# --------------------------------------------------------------------- rendering


def _names_clause(row: dict[str, Any]) -> str:
    common = row.get("common_name")
    head = f"{common} ({row['scientific_name']})" if common else row["scientific_name"]
    return f"{head} — USDA symbol {row['symbol']}"


def _native_clause(row: dict[str, Any]) -> str:
    labels = {
        "L48": "the lower 48 states",
        "AK": "Alaska",
        "HI": "Hawaii",
        "CAN": "Canada",
        "PR": "Puerto Rico",
        "VI": "the U.S. Virgin Islands",
        "SPM": "St. Pierre and Miquelon",
        "GL": "Greenland",
        "NAV": "Navassa Island",
        "CAN_L48": "North America",
    }
    native, introduced = [], []
    for status in row.get("native_status") or []:
        label = labels.get(status.get("Region"), status.get("Region"))
        if status.get("Type") == "Native":
            native.append(label)
        elif status.get("Type") == "Introduced":
            introduced.append(label)
    parts = []
    if native:
        parts.append(f"native to {_join(native)}")
    if introduced:
        parts.append(f"introduced in {_join(introduced)}")
    return "; ".join(parts)


def summary_text(row: dict[str, Any], traits: dict[str, str]) -> str:
    bits = [f"{_names_clause(row)}."]
    taxon = []
    if row.get("family"):
        taxon.append(f"family {row['family']}")
    if row.get("group_name"):
        taxon.append(row["group_name"].lower())
    habit = _join([h.lower() for h in row.get("growth_habit") or []])
    duration = _join([d.lower() for d in row.get("duration") or []])
    lead = " ".join(x for x in [duration, habit] if x)
    if lead:
        taxon.insert(0, f"a {lead}")
    if taxon:
        bits.append(f"{' — '.join([taxon[0]] + taxon[1:])}.".capitalize())
    native = _native_clause(row)
    if native:
        bits.append(f"It is {native}.")
    highlights = []
    if row.get("height_mature_ft"):
        highlights.append(f"mature height about {row['height_mature_ft']:g} ft")
    if row.get("shade_tolerance"):
        highlights.append(f"{row['shade_tolerance'].lower()} shade tolerance")
    if row.get("drought_tolerance"):
        highlights.append(f"{row['drought_tolerance'].lower()} drought tolerance")
    if row.get("ph_min") and row.get("ph_max"):
        highlights.append(f"soil pH {row['ph_min']:g}-{row['ph_max']:g}")
    if row.get("bloom_period"):
        highlights.append(f"blooms in {row['bloom_period'].lower()}")
    if row.get("toxicity"):
        highlights.append(f"toxicity: {row['toxicity'].lower()}")
    if highlights:
        bits.append(f"Key traits: {_join(highlights)}.")
    aliases = [a for a in (row.get("other_common_names") or []) if a][:6]
    if aliases:
        bits.append(f"Also known as {_join(aliases)}.")
    if traits.get("Commercial Availability"):
        bits.append(f"Commercial availability: {traits['Commercial Availability'].lower()}.")
    return " ".join(bits)


def growth_text(row: dict[str, Any], traits: dict[str, str]) -> str:
    bits = [f"Growth and form of {_names_clause(row)}:"]
    facts = []
    for label, key in (
        ("growth form", "Growth Form"),
        ("shape and orientation", "Shape and Orientation"),
        ("growth rate", "Growth Rate"),
        ("lifespan", "Lifespan"),
        ("active growth period", "Active Growth Period"),
        ("leaf retention", "Leaf Retention"),
        ("foliage color", "Foliage Color"),
        ("foliage texture", "Foliage Texture"),
        ("foliage porosity in summer", "Foliage Porosity Summer"),
        ("foliage porosity in winter", "Foliage Porosity Winter"),
        ("flower color", "Flower Color"),
        ("bloom period", "Bloom Period"),
        ("fruit or seed color", "Fruit/Seed Color"),
        ("fruit or seed abundance", "Fruit/Seed Abundance"),
        ("fruit or seed period", "Fruit/Seed Period Begin"),
        ("vegetative spread rate", "Vegetative Spread Rate"),
        ("seed spread rate", "Seed Spread Rate"),
        ("resprout ability", "Resprout Ability"),
        ("coppice potential", "Coppice Potential"),
    ):
        if traits.get(key):
            facts.append(f"{label} {traits[key].lower()}")
    if row.get("height_mature_ft"):
        facts.append(f"mature height {row['height_mature_ft']:g} ft")
    if traits.get("Height at 20 Years, Maximum (feet)"):
        facts.append(f"about {traits['Height at 20 Years, Maximum (feet)']} ft at 20 years")
    if row.get("root_depth_min_in"):
        facts.append(f"minimum root depth {row['root_depth_min_in']:g} in")
    bits.append(_join(facts) + ".")
    if _yes(traits, "Nitrogen Fixation") or (row.get("nitrogen_fixation") or "").lower() not in (
        "",
        "none",
    ):
        bits.append(f"Nitrogen fixation: {row.get('nitrogen_fixation', 'yes')}.")
    if _yes(traits, "Known Allelopath"):
        bits.append("It is a known allelopath.")
    return " ".join(bits)


def site_text(row: dict[str, Any], traits: dict[str, str]) -> str:
    bits = [f"Site conditions, tolerances and climate for {_names_clause(row)}:"]
    tolerances = []
    for label, key in (
        ("drought", "Drought Tolerance"),
        ("shade", "Shade Tolerance"),
        ("salinity", "Salinity Tolerance"),
        ("calcium carbonate (CaCO3)", "CaCO3 Tolerance"),
        ("anaerobic conditions", "Anaerobic Tolerance"),
        ("fire", "Fire Tolerance"),
        ("hedging", "Hedge Tolerance"),
    ):
        if traits.get(key):
            tolerances.append(f"{label} tolerance {traits[key].lower()}")
    if tolerances:
        bits.append(_join(tolerances).capitalize() + ".")
    if _yes(traits, "Fire Resistant"):
        bits.append("It is fire resistant.")
    soils = [
        name
        for name, key in (
            ("coarse", "Adapted to Coarse Textured Soils"),
            ("medium", "Adapted to Medium Textured Soils"),
            ("fine", "Adapted to Fine Textured Soils"),
        )
        if _yes(traits, key)
    ]
    soil_bits = []
    if soils:
        soil_bits.append(f"adapted to {_join(soils)} textured soils")
    if row.get("ph_min") and row.get("ph_max"):
        soil_bits.append(f"soil pH {row['ph_min']:g} to {row['ph_max']:g}")
    if traits.get("Fertility Requirement"):
        soil_bits.append(f"{traits['Fertility Requirement'].lower()} fertility requirement")
    if traits.get("Moisture Use"):
        soil_bits.append(f"{traits['Moisture Use'].lower()} moisture use")
    if soil_bits:
        bits.append(_join(soil_bits).capitalize() + ".")
    climate = []
    if row.get("temp_min_f") is not None:
        climate.append(f"minimum temperature {row['temp_min_f']:g} °F")
    if row.get("precip_min_in") and row.get("precip_max_in"):
        climate.append(f"annual precipitation {row['precip_min_in']:g}-{row['precip_max_in']:g} in")
    if row.get("frost_free_days_min"):
        climate.append(f"at least {row['frost_free_days_min']:g} frost-free days")
    if climate:
        bits.append(f"Climate: {_join(climate)}.")
    if row.get("wetland_status"):
        codes = {c.split(": ")[-1] for c in row["wetland_status"].split("; ")}
        readable = _join(sorted(WETLAND_CODES.get(c, c) for c in codes))
        bits.append(f"Wetland indicator status by region — {row['wetland_status']} ({readable}).")
    return " ".join(bits)


def propagation_text(row: dict[str, Any], traits: dict[str, str]) -> str:
    bits = [f"Propagation and establishment of {_names_clause(row)}:"]
    methods = [k.replace("Propagated by ", "").lower() for k in PROPAGATION_KEYS if _yes(traits, k)]
    if methods:
        bits.append(f"Propagated by {_join(methods)}.")
    else:
        bits.append("No propagation methods are recorded.")
    extra = []
    if _yes(traits, "Cold Stratification Required"):
        extra.append("cold stratification is required")
    if traits.get("Seedling Vigor"):
        extra.append(f"seedling vigor {traits['Seedling Vigor'].lower()}")
    if traits.get("Seed per Pound"):
        extra.append(f"about {traits['Seed per Pound']} seeds per pound")
    if traits.get("Planting Density per Acre, Minimum") and traits.get(
        "Planting Density per Acre, Maximum"
    ):
        extra.append(
            "planting density "
            f"{traits['Planting Density per Acre, Minimum']}-"
            f"{traits['Planting Density per Acre, Maximum']} per acre"
        )
    if traits.get("Fruit/Seed Persistence"):
        extra.append(f"fruit or seed persistence {traits['Fruit/Seed Persistence'].lower()}")
    if traits.get("After Harvest Regrowth Rate"):
        extra.append(f"after-harvest regrowth rate {traits['After Harvest Regrowth Rate'].lower()}")
    if extra:
        bits.append(_join(extra).capitalize() + ".")
    if traits.get("Commercial Availability"):
        bits.append(f"Commercial availability: {traits['Commercial Availability'].lower()}.")
    return " ".join(bits)


def uses_text(row: dict[str, Any], traits: dict[str, str], wildlife: dict[str, Any]) -> str:
    bits = [f"Uses, wildlife value and safety for {_names_clause(row)}:"]
    if traits.get("Toxicity"):
        bits.append(f"Recorded toxicity: {traits['Toxicity'].lower()}.")
    palat = []
    for label, key in (
        ("humans", "Palatable Human"),
        ("browsing animals", "Palatable Browse Animal"),
        ("grazing animals", "Palatable Graze Animal"),
    ):
        if traits.get(key):
            palat.append(f"palatability to {label} {traits[key].lower()}")
    if traits.get("Protein Potential"):
        palat.append(f"protein potential {traits['Protein Potential'].lower()}")
    if palat:
        bits.append(_join(palat).capitalize() + ".")
    products = [k.replace(" Product", "").lower() for k in PRODUCT_KEYS if _yes(traits, k)]
    if products:
        bits.append(f"Suitable for {_join(products)} products.")
    food = [f.get("CommonName") or f.get("Name") for f in (wildlife.get("Food") or [])]
    cover = [c.get("CommonName") or c.get("Name") for c in (wildlife.get("Cover") or [])]
    if food:
        bits.append(f"Provides food for {_join([f for f in food if f][:8])}.")
    if cover:
        bits.append(f"Provides cover for {_join([c for c in cover if c][:8])}.")
    if not food and not cover:
        bits.append("No wildlife food or cover values are recorded.")
    return " ".join(bits)


def filter_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Payload used for Qdrant filtering and for citation rendering."""
    return {
        "species_id": row["id"],
        "symbol": row["symbol"],
        "scientific_name": row["scientific_name"],
        "scientific_no_author": row["scientific_no_author"],
        "common_name": row["common_name"],
        "family": row["family"],
        "genus": row["genus"],
        "growth_habit": row["growth_habit"],
        "duration": row["duration"],
        "native_regions": [
            s.get("Region") for s in (row.get("native_status") or []) if s.get("Type") == "Native"
        ],
        "ph_min": row["ph_min"],
        "ph_max": row["ph_max"],
        "height_mature_ft": row["height_mature_ft"],
        "temp_min_f": row["temp_min_f"],
        "precip_min_in": row["precip_min_in"],
        "precip_max_in": row["precip_max_in"],
        "drought_tolerance": row["drought_tolerance"],
        "shade_tolerance": row["shade_tolerance"],
        "salinity_tolerance": row["salinity_tolerance"],
        "fire_tolerance": row["fire_tolerance"],
        "moisture_use": row["moisture_use"],
        "growth_rate": row["growth_rate"],
        "lifespan": row["lifespan"],
        "bloom_period": row["bloom_period"],
        "flower_color": row["flower_color"],
        "toxicity": row["toxicity"],
        "nitrogen_fixation": row["nitrogen_fixation"],
        "palatable_human": row["palatable_human"],
        "wetland_status": row["wetland_status"],
        "profile_url": row["profile_url"],
    }


def build_documents(row: dict[str, Any], wildlife: dict[str, Any]) -> list[dict[str, Any]]:
    traits: dict[str, str] = row.get("raw_characteristics") or {}
    metadata = filter_metadata(row)
    title = _names_clause(row)
    renderers = {
        "summary": summary_text(row, traits),
        "growth": growth_text(row, traits),
        "site": site_text(row, traits),
        "propagation": propagation_text(row, traits),
        "uses": uses_text(row, traits, wildlife),
    }
    docs = []
    for section, text in renderers.items():
        clean = " ".join(text.split())
        docs.append(
            {
                "doc_id": f"{row['id']}::{section}",
                "species_id": row["id"],
                "section": section,
                "chunk_index": 0,
                "title": title,
                "text": clean,
                "content_hash": hashlib.sha256(clean.encode()).hexdigest()[:16],
                "metadata": {**metadata, "section": section},
            }
        )
    return docs
