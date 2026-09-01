"""Typed, cached client for the USDA PLANTS Services API.

The API needs no authentication. It publishes an OpenAPI spec at
``/swagger/v1/swagger.json``. Every response is cached to ``data/cache`` so
re-running ingestion is offline, fast, and reproducible even if the service is
unavailable.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from urllib3.util.retry import Retry

from app.config import get_settings

USDA_HOST = "https://plantsservices.sc.egov.usda.gov"
# PDFs live on the public site, not on the services host, which 404s for them.
USDA_DOCS_HOST = "https://plants.usda.gov"
PROFILE_URL = "https://plants.usda.gov/plant-profile/{symbol}"
TAG_RE = re.compile(r"<[^>]+>")


def strip_tags(value: str | None) -> str | None:
    """USDA embeds ``<i>`` markup in scientific names."""
    if not value:
        return value
    return TAG_RE.sub("", value).strip()


class UsdaClient:
    def __init__(self, cache_dir: Path | None = None, timeout: int = 60) -> None:
        settings = get_settings()
        self.base = settings.usda_api_base.rstrip("/")
        self.timeout = timeout
        self.cache_dir = cache_dir or settings.cache_dir
        self.session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=4,
                backoff_factor=1.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            ),
            pool_maxsize=settings.usda_max_workers * 2,
        )
        self.session.mount("https://", adapter)
        self.session.headers["User-Agent"] = "plant-rag/0.1 (LLM Zoomcamp project)"

    # ---------------------------------------------------------------- caching

    def _cache_path(self, key: str) -> Path:
        safe = key.strip("/").replace("/", "__").replace("?", "_").replace("=", "-")
        return self.cache_dir / f"{safe}.json"

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=2, max=30),
        reraise=True,
    )
    def _fetch(self, path: str) -> Any:
        response = self.session.get(f"{self.base}/{path.lstrip('/')}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get(self, path: str, use_cache: bool = True) -> Any:
        cache_file = self._cache_path(path)
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())
        payload = self._fetch(path)
        cache_file.write_text(json.dumps(payload))
        return payload

    # ---------------------------------------------------------------- endpoints

    def characterized_species(self) -> list[dict[str, Any]]:
        """Every species that has characteristics data (~2,186 records)."""
        rows = self.get("characteristicSearchResults")
        return [r for r in rows if not r.get("isSynonym")]

    def characteristics(self, species_id: int) -> list[dict[str, Any]]:
        return self.get(f"PlantCharacteristics/{species_id}") or []

    def profile(self, species_id: int) -> dict[str, Any]:
        return self.get(f"PlantProfile/{species_id}") or {}

    def wetland(self, species_id: int) -> list[dict[str, Any]]:
        return self.get(f"PlantWetland/{species_id}") or []

    def wildlife(self, species_id: int) -> dict[str, Any]:
        return self.get(f"PlantWildlife/{species_id}") or {}

    def ethnobotany(self, species_id: int) -> list[dict[str, Any]]:
        return self.get(f"PlantEthnobotany/{species_id}") or []

    def search(self, text: str) -> list[dict[str, Any]]:
        """Name search across the full PLANTS taxonomy (not just the corpus)."""
        return (
            self.get(f"PlantSearch?searchText={requests.utils.quote(text)}", use_cache=False) or []
        )

    def vocabulary(self, kind: str) -> Any:
        """Controlled vocabularies: GrowthHabitSearch, DurationSearch, StateSearch, ..."""
        return self.get(kind)

    def pdf(self, url_path: str) -> bytes | None:
        """Download a fact-sheet / plant-guide PDF, cached on disk."""
        settings = get_settings()
        target = settings.pdf_dir / Path(url_path).name
        if target.exists():
            return target.read_bytes()
        try:
            url = url_path if url_path.startswith("http") else f"{USDA_DOCS_HOST}{url_path}"
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException:
            return None
        if not response.content.startswith(b"%PDF"):
            return None
        target.write_bytes(response.content)
        return response.content

    # ---------------------------------------------------------------- bulk

    def hydrate(
        self, species_ids: list[int], workers: int | None = None
    ) -> dict[int, dict[str, Any]]:
        """Fetch characteristics + profile + wetland + wildlife for many species."""
        workers = workers or get_settings().usda_max_workers

        def one(species_id: int) -> tuple[int, dict[str, Any]]:
            record = {
                "characteristics": self.characteristics(species_id),
                "profile": self.profile(species_id),
                "wetland": self.wetland(species_id),
                "wildlife": self.wildlife(species_id),
            }
            return species_id, record

        out: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for species_id, record in pool.map(one, species_ids):
                out[species_id] = record
        return out


def polite_sleep(seconds: float = 0.05) -> None:
    time.sleep(seconds)
