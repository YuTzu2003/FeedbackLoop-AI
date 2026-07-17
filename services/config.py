import os
from dataclasses import dataclass


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: int
    llm_max_tokens: int
    llm_temperature: float
    ollama_base_url: str
    embedding_model: str
    embedding_timeout: int
    weaviate_host: str
    weaviate_port: int
    weaviate_grpc_port: int


def load_settings() -> Settings:
    return Settings(
        llm_base_url=required_env("LLM_BASE_URL"),
        llm_api_key=required_env("LLM_API_KEY"),
        llm_model=required_env("LLM_MODEL"),
        llm_timeout=int(required_env("LLM_TIMEOUT")),
        llm_max_tokens=int(required_env("LLM_MAX_TOKENS")),
        llm_temperature=float(required_env("LLM_TEMPERATURE")),
        ollama_base_url=required_env("OLLAMA_BASE_URL").rstrip("/"),
        embedding_model=required_env("EMBEDDING_MODEL"),
        embedding_timeout=int(required_env("EMBEDDING_TIMEOUT")),
        weaviate_host=required_env("WEAVIATE_HOST"),
        weaviate_port=int(required_env("WEAVIATE_PORT")),
        weaviate_grpc_port=int(required_env("WEAVIATE_GRPC_PORT")),
    )
