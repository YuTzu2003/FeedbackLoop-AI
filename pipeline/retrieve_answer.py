import openai
import re
from collections import defaultdict
from weaviate.classes.query import Filter, MetadataQuery
from services.api import LLMSettings, llm_client, get_rag_prompt
from services.config import Settings
from services.vectordb import RagServiceError, embedding, rag_collection, weaviate_client

STRUCTURED_QUERY_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)+\b|\b\d{4}/\d[A-Za-z]?\b|\b(?:HSIL|CIN2-3)\b|欄位|表格|代碼|分期|診斷碼|申報欄位|資料欄位",
    re.IGNORECASE,
)


def build_search_queries(question: str, llm_settings: LLMSettings) -> list[str]:
    prompt = (
        "請將下列問題改寫成三條 PDF 檢索查詢，每行一條，不要編號或說明。\n"
        "保留欄位編號、章節編號、表格列、診斷或分期代碼；必要時加入中英文同義詞。\n"
        "查詢必須聚焦於文件中的原始文字，不要回答問題，也不要引入文件未提及的年份或規則。\n\n"
        f"原始問題：{question}"
    )
    client = llm_client(llm_settings)
    try:
        response = client.chat.completions.create(
            model=llm_settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
    except openai.OpenAIError:
        return [question]

    queries = [question]
    seen = {question.casefold()}
    for line in content.splitlines():
        rewrite = re.sub(r"^(?:[-•]|\d+[.)])\s*", "", line).strip()
        normalized = rewrite.casefold()
        if normalized and normalized not in seen:
            queries.append(rewrite)
            seen.add(normalized)
        if len(queries) == 4:
            break
    return queries


def select_hybrid_alpha(question: str) -> float:
    return 0.25 if STRUCTURED_QUERY_PATTERN.search(question) else 0.65


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
        for document_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def retrieve_chunks(question: str,document_id: str,settings: Settings,llm_settings: LLMSettings,search_mode: str,) -> list[dict]:
    queries = build_search_queries(question, llm_settings)
    client = weaviate_client(settings)
    try:
        collection = rag_collection(client)
        result_groups = []
        hybrid_alpha = select_hybrid_alpha(question)
        for query in queries:
            if search_mode == "hybrid":
                response = collection.query.hybrid(
                    query=query,
                    vector=embedding(query, settings),
                    alpha=hybrid_alpha,
                    filters=Filter.by_property("document_id").equal(document_id),
                    limit=settings.retrieval_top_k,
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
                    limit=settings.retrieval_top_k,
                    return_metadata=MetadataQuery(distance=True),
                )
                result_groups.append([
                    {"uuid": str(item.uuid), **item.properties, "score": 1 - (item.metadata.distance or 0)}
                    for item in response.objects
                ])
        candidates = reciprocal_rank_fusion(result_groups)[:settings.rrf_candidate_top_k]
        return candidates[:settings.final_context_top_k]
    finally:
        client.close()


def answer_from_chunks(question: str, chunks: list[dict], llm_settings: LLMSettings, personal_instruction: str = "") -> str:
    contexts = "\n\n".join(f"[{item['title']} | {item['url']}]\n{item['content']}" for item in chunks)
    prompt = get_rag_prompt(question, contexts, personal_instruction)
    client = llm_client(llm_settings)
    try:
        response = client.chat.completions.create(
            model=llm_settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_settings.temperature,
        )
        content = response.choices[0].message.content
        return content.strip()
    except openai.OpenAIError as error:
        raise RagServiceError(f"模型回應失敗: {error}", 500) from error


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
    except openai.OpenAIError as error:
        raise RagServiceError(f"模型回應失敗: {error}", 500) from error
