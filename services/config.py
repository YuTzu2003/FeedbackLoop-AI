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
    elasticsearch_url: str
    elasticsearch_index: str
    elasticsearch_api_key: str | None = None
    elasticsearch_username: str | None = None
    elasticsearch_password: str | None = None
    embedding_timeout: int = 30
    reranker_base_url: str | None = None
    reranker_model: str | None = None
    reranker_max_query_chars: int = 200
    reranker_max_document_chars: int = 800
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
        elasticsearch_url=required_env("ELASTICSEARCH_URL"),
        elasticsearch_index=os.getenv("ELASTICSEARCH_INDEX", "feedbackloop-documents"),
        elasticsearch_api_key=os.getenv("ELASTICSEARCH_API_KEY"),
        elasticsearch_username=os.getenv("ELASTICSEARCH_USERNAME"),
        elasticsearch_password=os.getenv("ELASTICSEARCH_PASSWORD"),
        reranker_base_url=os.getenv("RERANKER_BASE_URL"),
        reranker_model=os.getenv("RERANKER_MODEL"),
        reranker_max_query_chars=int(os.getenv("RERANKER_MAX_QUERY_CHARS", "200")),
        reranker_max_document_chars=int(os.getenv("RERANKER_MAX_DOCUMENT_CHARS", "800")),
        retrieval_top_k=int(os.getenv("RAG_RETRIEVAL_TOP_K", "20")),
        rrf_candidate_top_k=int(os.getenv("RAG_RRF_CANDIDATE_TOP_K", "30")),
        rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "8")),
        final_context_top_k=int(os.getenv("RAG_FINAL_CONTEXT_TOP_K", "8")),
        field_summary_candidate_k=int(os.getenv("RAG_FIELD_SUMMARY_CANDIDATE_K", "8")),
        field_detail_candidate_k=int(os.getenv("RAG_FIELD_DETAIL_CANDIDATE_K", "20")),
        general_candidate_k=int(os.getenv("RAG_GENERAL_CANDIDATE_K", "12")),
    )
