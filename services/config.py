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


def load_settings() -> Settings:
    return Settings(
        embedding_base_url=required_env("EMBEDDING_BASE_URL"),
        embedding_model=required_env("EMBEDDING_MODEL"),
        weaviate_host=required_env("WEAVIATE_HOST"),
        weaviate_port=int(required_env("WEAVIATE_PORT")),
        weaviate_grpc_port=int(required_env("WEAVIATE_GRPC_PORT")),)