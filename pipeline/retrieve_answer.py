import openai
from weaviate.classes.query import Filter, MetadataQuery
from services.api import LLMSettings, llm_client, get_rag_prompt
from services.config import Settings
from services.vectordb import RagServiceError, TOP_K, embedding, rag_collection, weaviate_client

def retrieve_chunks(question: str, document_id: str, settings: Settings) -> list[dict]:
    client = weaviate_client(settings)
    try:
        response = rag_collection(client).query.near_vector(
            near_vector=embedding(question, settings),
            filters=Filter.by_property("document_id").equal(document_id),
            limit=TOP_K,
            return_metadata=MetadataQuery(distance=True),
        )
        return [{**item.properties, "score": 1 - (item.metadata.distance or 0)} for item in response.objects]
    finally:
        client.close()


def answer_from_chunks(question: str, chunks: list[dict], llm_settings: LLMSettings) -> str:
    contexts = "\n\n".join(f"[{item['title']} | {item['url']}]\n{item['content']}" for item in chunks)
    prompt = get_rag_prompt(question, contexts)
    client = llm_client(llm_settings)
    try:
        response = client.chat.completions.create(
            model=llm_settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_settings.temperature,
        )
        content = response.choices[0].message.content
        return content.strip()
    except openai.APITimeoutError as error:
        raise RagServiceError("模型回應逾時，請稍後再試。", 504) from error


def answer_from_history(messages: list[dict], llm_settings: LLMSettings) -> str:
    client = llm_client(llm_settings)
    try:
        response = client.chat.completions.create(
            model=llm_settings.model,
            messages=messages,
            temperature=llm_settings.temperature,
        )
        content = response.choices[0].message.content
        return content.strip()
    except openai.APITimeoutError as error:
        raise RagServiceError("模型回應逾時，請稍後再試。", 504) from error
