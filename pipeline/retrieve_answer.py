import openai
from collections import defaultdict
from weaviate.classes.query import Filter, MetadataQuery
from services.api import LLMSettings, llm_client, get_rag_prompt
from services.config import Settings
from services.vectordb import RagServiceError, TOP_K, embedding, rag_collection, weaviate_client

def generate_three_queries(question: str, llm_settings: LLMSettings) -> list[str]:
    prompt = (
        "Generate three different search queries for the following question. "
        "Return one query per line.\n\n"
        f"Question: {question}"
    )
    client = llm_client(llm_settings)
    try:
        response = client.chat.completions.create(
            model=llm_settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        queries = [line.strip() for line in content.splitlines() if line.strip()]
        return queries[:3] or [question]
    except openai.APITimeoutError as error:
        raise RagServiceError("LLM request timed out", 504) from error


def reciprocal_rank_fusion(result_groups: list[list[dict]], k: int = 60) -> list[dict]:
    scores = defaultdict(float)
    documents = {}
    for documents_in_group in result_groups:
        for rank, document in enumerate(documents_in_group, start=1):
            document_id = document["uuid"]
            scores[document_id] += 1 / (k + rank)
            documents[document_id] = document
    return [
        {**documents[document_id], "rrf_score": score}
        for document_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def retrieve_chunks(
    question: str,
    document_id: str,
    settings: Settings,
    llm_settings: LLMSettings,
    search_mode: str,
) -> list[dict]:
    queries = generate_three_queries(question, llm_settings)
    client = weaviate_client(settings)
    try:
        collection = rag_collection(client)
        result_groups = []
        for query in queries:
            if search_mode == "hybrid":
                response = collection.query.hybrid(
                    query=query,
                    vector=embedding(query, settings),
                    alpha=0.5,
                    filters=Filter.by_property("document_id").equal(document_id),
                    limit=TOP_K,
                    return_metadata=MetadataQuery(score=True),
                )
                result_groups.append([
                    {"uuid": str(item.uuid), **item.properties, "score": item.metadata.score}
                    for item in response.objects
                ])
            else:
                response = collection.query.near_vector(
                    near_vector=embedding(query, settings),
                    filters=Filter.by_property("document_id").equal(document_id),
                    limit=TOP_K,
                    return_metadata=MetadataQuery(distance=True),
                )
                result_groups.append([
                    {"uuid": str(item.uuid), **item.properties, "score": 1 - (item.metadata.distance or 0)}
                    for item in response.objects
                ])
        return reciprocal_rank_fusion(result_groups)[:TOP_K]
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
