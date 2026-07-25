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


FIELD_DETAIL_TERMS = ("欄位敘述", "收錄目的", "編碼指引", "icd-o", "m-code", "solid tumor", "規則")
FIELD_FULL_TERMS = ("完整", "全部", "所有資訊", "完整介紹")
FIELD_BLOCK_ORDER = {"field_summary": 0, "field_detail_parent": 1, "field_detail_part": 2}
DETAIL_TYPE_ORDER = {"description": 0, "purpose": 1, "coding_instruction": 2, "note": 3, "exception": 4}
FIELD_NAME_PREFIXES = ("請完整說明", "請完整介紹", "完整說明", "完整介紹", "請幫我說明", "請說明", "說明", "介紹", "關於")


def classify_field_intent(question: str) -> str:
    normalized = question.casefold()
    if not ("欄位" in question or "histology" in normalized or "組織型態" in question):
        return "general"
    if any(term in normalized for term in FIELD_FULL_TERMS):
        return "full"
    if any(term in normalized for term in FIELD_DETAIL_TERMS) or "說明" in question:
        return "detail"
    return "basic"


def unique_queries(queries: list[str]) -> list[str]:
    unique = []
    seen = set()
    for query in queries:
        normalized = query.casefold()
        if normalized and normalized not in seen:
            unique.append(query)
            seen.add(normalized)
    return unique


def extract_field_name(question: str) -> str:
    if "欄位" not in question:
        return ""
    field_name = question.split("欄位", maxsplit=1)[0].strip(" ：:，,。？?的")
    for prefix in FIELD_NAME_PREFIXES:
        if field_name.startswith(prefix):
            field_name = field_name[len(prefix) :].strip(" ：:，,。？?的")
            break
    return field_name


def block_filter(document_id: str, block_types: tuple[str, ...] = ()):
    filters = Filter.by_property("document_id").equal(document_id)
    if not block_types:
        return filters
    type_filter = Filter.by_property("block_type").equal(block_types[0])
    for block_type in block_types[1:]:
        type_filter = type_filter | Filter.by_property("block_type").equal(block_type)
    return filters & type_filter


def search_result_groups(collection, queries: list[str], document_id: str, settings: Settings, search_mode: str, limit: int, block_types: tuple[str, ...] = (), hybrid_alpha: float | None = None) -> list[list[dict]]:
    result_groups = []
    filters = block_filter(document_id, block_types)
    for query in queries:
        if search_mode == "hybrid":
            response = collection.query.hybrid(
                query=query,
                vector=embedding(query, settings),
                alpha=hybrid_alpha if hybrid_alpha is not None else select_hybrid_alpha(query),
                filters=filters,
                limit=limit,
                return_metadata=MetadataQuery(score=True),
            )
            result_groups.append([
                {"uuid": str(item.uuid), **item.properties, "score": item.metadata.score}
                for item in response.objects
            ])
        else:
            response = collection.query.near_vector(
                near_vector=embedding(query, settings),
                filters=filters,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )
            result_groups.append([
                {"uuid": str(item.uuid), **item.properties, "score": 1 - (item.metadata.distance or 0)}
                for item in response.objects
            ])
    return result_groups


def build_field_detail_queries(summary_chunks: list[dict]) -> list[str]:
    queries = []
    for chunk in summary_chunks:
        field_name = str(chunk.get("field_name") or "").strip()
        field_code = str(chunk.get("field_code") or "").strip()
        english_name = str(chunk.get("field_english_name") or "").strip()
        if not field_name:
            continue
        queries.extend(
            [
                f"{field_name} 欄位敘述",
                f"{field_name} 收錄目的 編碼指引",
                f"癌登欄位序號 {field_code}",
                f"{field_name} {english_name}",
            ]
        )
    return unique_queries(queries)


def fetch_field_expansion(collection, document_id: str, field_codes: set[str]) -> list[dict]:
    chunks = []
    seen = set()
    for field_code in sorted(field_codes):
        response = collection.query.fetch_objects(
            filters=block_filter(document_id) & Filter.by_property("field_code").equal(field_code),
            limit=50,
        )
        for item in response.objects:
            chunk = {"uuid": str(item.uuid), **item.properties}
            chunk_id = chunk.get("chunk_id") or chunk["uuid"]
            if chunk_id not in seen and chunk.get("field_code") == field_code and chunk.get("block_type") in FIELD_BLOCK_ORDER:
                chunks.append(chunk)
                seen.add(chunk_id)
    return chunks


def fetch_exact_field_summaries(collection, document_id: str, field_name: str) -> list[dict]:
    if not field_name:
        return []
    response = collection.query.fetch_objects(
        filters=block_filter(document_id, ("field_summary",)) & Filter.by_property("field_name").equal(field_name),
        limit=5,
    )
    return [
        {"uuid": str(item.uuid), **item.properties}
        for item in response.objects
        if item.properties.get("block_type") == "field_summary" and item.properties.get("field_name") == field_name
    ]


def order_field_chunks(candidates: list[dict], expanded_chunks: list[dict], limit: int) -> list[dict]:
    ordered_expansion = sorted(
        expanded_chunks,
        key=lambda chunk: (
            str(chunk.get("field_code") or ""),
            FIELD_BLOCK_ORDER.get(chunk.get("block_type"), 99),
            DETAIL_TYPE_ORDER.get(chunk.get("detail_type"), 99),
        ),
    )
    combined = []
    seen = set()
    for chunk in [*ordered_expansion, *candidates]:
        chunk_id = chunk.get("chunk_id") or chunk.get("uuid")
        if chunk_id and chunk_id not in seen:
            combined.append(chunk)
            seen.add(chunk_id)
        if len(combined) == limit:
            break
    return combined


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
        intent = classify_field_intent(question)
        hybrid_alpha = select_hybrid_alpha(question)
        if intent == "general":
            result_groups = search_result_groups(
                collection, queries, document_id, settings, search_mode, settings.retrieval_top_k, hybrid_alpha=hybrid_alpha
            )
            candidates = reciprocal_rank_fusion(result_groups)[:settings.rrf_candidate_top_k]
            return candidates[:settings.final_context_top_k]

        summary_groups = search_result_groups(
            collection, queries, document_id, settings, search_mode, settings.field_summary_candidate_k, ("field_summary",), hybrid_alpha
        )
        exact_summary_chunks = fetch_exact_field_summaries(collection, document_id, extract_field_name(question))
        summary_chunks = order_field_chunks(reciprocal_rank_fusion(summary_groups), exact_summary_chunks, settings.rrf_candidate_top_k)
        detail_queries = unique_queries([question, *build_field_detail_queries(summary_chunks)])
        detail_groups = search_result_groups(
            collection,
            detail_queries,
            document_id,
            settings,
            search_mode,
            settings.field_detail_candidate_k,
            ("field_detail_parent", "field_detail_part"),
            hybrid_alpha,
        )
        general_groups = search_result_groups(
            collection, queries, document_id, settings, search_mode, settings.general_candidate_k, ("general_text",), hybrid_alpha
        )
        result_groups = [*summary_groups, *detail_groups, *general_groups]
        candidates = reciprocal_rank_fusion(result_groups)[:settings.rrf_candidate_top_k]
        field_codes = {str(chunk.get("field_code")) for chunk in exact_summary_chunks if chunk.get("field_code")}
        if not field_codes:
            field_codes = {str(chunk.get("field_code")) for chunk in [*summary_chunks, *candidates] if chunk.get("field_code")}
        if intent in {"detail", "full"} and field_codes:
            expanded_chunks = fetch_field_expansion(collection, document_id, field_codes)
            if expanded_chunks:
                return order_field_chunks([], expanded_chunks, settings.final_context_top_k)
        if intent == "basic" and exact_summary_chunks:
            return exact_summary_chunks[:settings.final_context_top_k]
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
