from dataclasses import dataclass
import openai
from services.config import required_env

@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout: int = 300
    max_tokens: int = 8192
    temperature: float = 0.7

def load_llm_settings() -> LLMSettings:
    return LLMSettings(
        base_url=required_env("LLM_BASE_URL"),
        api_key=required_env("LLM_API_KEY"),
        model=required_env("LLM_MODEL"),
    )

def llm_client(settings: LLMSettings) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout,
    )

def get_rag_prompt(question: str, contexts: str) -> str:
    return (
        "Please answer the questions clearly in respond to the question in the language the user used, based on the following sources.\n"
        "If the source content is insufficient to fully answer the question, you must explicitly reply: 'Based on the currently provided information, I cannot answer this question.' Under no circumstances should you return an empty response.\n\n"
        f"Question: {question}\n\nSources:\n{contexts}"
    )

def get_system_prompt() -> dict:
    return {
        "role": "system",
        "content": (
            "You are a retrieval-augmented generation (RAG) assistant. "
            "Answer the user's question accurately and concisely using the provided context and conversation history as the primary source of truth. "
            "If the available information is insufficient, clearly state that you do not have enough information to answer; do not invent facts, sources, or citations. "
            "Keep responses well-structured, factual, and directly responsive. Never return an empty or meaningless response."
        ),
    }