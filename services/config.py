import os
from dataclasses import dataclass

def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

@dataclass(frozen=True)
class Settings:
    embedding_base_url: str
    embedding_model: str
    weaviate_host: str
    weaviate_port: int
    weaviate_grpc_port: int
    embedding_timeout: int = 30
    retrieval_top_k: int = 20
    rrf_candidate_top_k: int = 30
    rerank_top_k: int = 8
    final_context_top_k: int = 8
    field_summary_candidate_k: int = 8
    field_detail_candidate_k: int = 20
    general_candidate_k: int = 12


def load_settings() -> Settings:
    return Settings(
        embedding_base_url=required_env("EMBEDDING_BASE_URL"),
        embedding_model=required_env("EMBEDDING_MODEL"),
        weaviate_host=required_env("WEAVIATE_HOST"),
        weaviate_port=int(required_env("WEAVIATE_PORT")),
        weaviate_grpc_port=int(required_env("WEAVIATE_GRPC_PORT")),
        retrieval_top_k=int(os.getenv("RAG_RETRIEVAL_TOP_K", "20")),
        rrf_candidate_top_k=int(os.getenv("RAG_RRF_CANDIDATE_TOP_K", "30")),
        rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "8")),
        final_context_top_k=int(os.getenv("RAG_FINAL_CONTEXT_TOP_K", "8")),
        field_summary_candidate_k=int(os.getenv("RAG_FIELD_SUMMARY_CANDIDATE_K", "8")),
        field_detail_candidate_k=int(os.getenv("RAG_FIELD_DETAIL_CANDIDATE_K", "20")),
        general_candidate_k=int(os.getenv("RAG_GENERAL_CANDIDATE_K", "12")),
    )
