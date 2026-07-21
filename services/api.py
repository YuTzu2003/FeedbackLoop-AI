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
        "請根據下列來源，以繁體中文清楚回答問題。\n"
        "若來源內容不足以完整回答，請務必明確回覆：「根據目前提供的資料，我無法回答這個問題。」，絕對不可以回傳空白。\n\n"
        f"問題：{question}\n\n來源：\n{contexts}"
    )


def get_system_prompt() -> dict:
    return {"role": "system", "content": "你是一個樂於助人的 AI 助手。請始終以繁體中文清晰、簡明地回答問題，絕對不可以回傳空白或無意義的內容。"}
