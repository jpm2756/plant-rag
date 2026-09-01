from app.tools import build_where
from app.vectorstore import build_filter, fuse_rrf
from ingestion.factsheets import chunk_text, extract_text


def _hit(doc_id, rank, arm):
    return {"doc_id": doc_id, "rank": rank, "score": 1.0, "arms": [arm], "species_id": 1}


def test_fuse_rrf_rewards_documents_found_by_both_arms():
    sparse = [_hit("a", 1, "sparse"), _hit("b", 2, "sparse")]
    dense = [_hit("c", 1, "dense"), _hit("b", 2, "dense")]
    fused = fuse_rrf([sparse, dense], k=10)
    assert fused[0]["doc_id"] == "b"
    assert sorted(fused[0]["arms"]) == ["dense", "sparse"]
    assert [h["rank"] for h in fused] == [1, 2, 3]


def test_build_filter_handles_scalars_lists_ranges_and_noise():
    assert build_filter(None) is None
    assert build_filter({"family": "", "toxicity": None}) is None
    qdrant_filter = build_filter(
        {
            "growth_habit": "Shrub",
            "shade_tolerance": ["Tolerant", "Intermediate"],
            "height_mature_ft": {"lte": 6},
        }
    )
    assert len(qdrant_filter.must) == 3


def test_build_where_generates_parameterised_sql():
    where, params = build_where(
        {"growth_habit": ["Shrub", "Tree"], "height_mature_ft": {"lte": 6}, "bogus": "x"}
    )
    assert where.count("%s") == 3
    assert params == ("Shrub", "Tree", 6.0)
    assert "height_mature_ft <= %s" in where


def test_build_where_defaults_to_true_when_no_usable_filters():
    assert build_where({}) == ("TRUE", ())


def test_chunk_text_splits_long_text_with_overlap_and_drops_scraps():
    text = "\n\n".join(f"Paragraph {i}. " + "botanical prose " * 60 for i in range(6))
    chunks = chunk_text(text, target_chars=1200, overlap_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1400 for c in chunks)
    assert chunk_text("too short") == []


def test_extract_text_survives_a_real_pdf(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    target = tmp_path / "blank.pdf"
    with target.open("wb") as fh:
        writer.write(fh)
    assert extract_text(target.read_bytes()) == ""
