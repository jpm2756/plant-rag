from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_judge_model: str = "gpt-4o-mini"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "plantrag"
    postgres_password: str = "plantrag"
    postgres_db: str = "plantrag"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "plants"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_mode: str = "hybrid_rerank_dual"
    filter_mode: str = "soft"
    use_rewrite: int = 1
    top_k: int = 5
    candidate_k: int = 30
    rrf_k: int = 60
    prompt_variant: str = "v1"

    usda_api_base: str = "https://plantsservices.sc.egov.usda.gov/api"
    usda_max_workers: int = 4
    ingest_limit: int = 0
    include_factsheets: int = 1

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cache_dir(self) -> Path:
        d = DATA_DIR / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def pdf_dir(self) -> Path:
        d = DATA_DIR / "pdf"
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()
