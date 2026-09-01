"""Fact-sheet / plant-guide PDF stage.

USDA publishes per-species fact sheets and plant guides as PDFs (e.g.
``/DocumentLibrary/factsheet/pdf/fs_abba.pdf``). They are genuine long-form prose
and make the corpus a mix of structured traits and unstructured text.
"""

from __future__ import annotations

import hashlib
import io
import re
from typing import Any

from pypdf import PdfReader

from ingestion.cards import filter_metadata
from ingestion.usda_client import UsdaClient

BOILERPLATE = re.compile(
    r"(USDA NRCS National Plant Data (Team|Center)|Plant Fact Sheet|Plant Guide|"
    r"The U\.S\. Department of Agriculture \(USDA\) prohibits|Federal Relay Service|"
    r"Helping People Help the Land|www\.plants\.usda\.gov)",
    re.IGNORECASE,
)
WHITESPACE = re.compile(r"[ \t]+")


def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a bad page shouldn't kill ingestion
            continue
    text = "\n".join(pages)
    lines = [
        WHITESPACE.sub(" ", line).strip()
        for line in text.splitlines()
        if line.strip() and not BOILERPLATE.search(line)
    ]
    return "\n".join(lines)


def chunk_text(text: str, target_chars: int = 1800, overlap_chars: int = 300) -> list[str]:
    """Paragraph-aware chunking (~450 tokens with ~75 tokens of overlap)."""
    paragraphs = [
        p.strip() for p in re.split(r"\n{2,}|\n(?=[A-Z][A-Za-z ]{3,40}\n)", text) if p.strip()
    ]
    if not paragraphs:
        paragraphs = [text]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= target_chars:
            current = f"{current}\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= target_chars:
            tail = current[-overlap_chars:] if current else ""
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph
        else:
            for start in range(0, len(paragraph), target_chars - overlap_chars):
                chunks.append(paragraph[start : start + target_chars])
            current = ""
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) > 200]


def documents_for_species(
    client: UsdaClient, row: dict[str, Any], max_chunks: int = 12
) -> list[dict[str, Any]]:
    urls = list(row.get("factsheet_urls") or []) + list(row.get("plantguide_urls") or [])
    if not urls:
        return []
    metadata = filter_metadata(row)
    docs: list[dict[str, Any]] = []
    index = 0
    for url in urls:
        pdf_bytes = client.pdf(url)
        if not pdf_bytes:
            continue
        kind = "plantguide" if "plantguide" in url else "factsheet"
        text = extract_text(pdf_bytes)
        for chunk in chunk_text(text)[:max_chunks]:
            docs.append(
                {
                    "doc_id": f"{row['id']}::{kind}::{index}",
                    "species_id": row["id"],
                    "section": kind,
                    "chunk_index": index,
                    "title": f"{row.get('common_name') or row['scientific_name']} — USDA {kind}",
                    "text": chunk,
                    "content_hash": hashlib.sha256(chunk.encode()).hexdigest()[:16],
                    "metadata": {**metadata, "section": kind, "source_url": url},
                }
            )
            index += 1
    return docs
