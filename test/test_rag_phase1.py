from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch
from pathlib import Path

import openai

from pipeline import retrieve_answer
from services.api import LLMSettings
from services.config import Settings, load_settings
from test.evaluate_manual_retrieval import evaluate_cases, load_cases


def llm_settings() -> LLMSettings:
    return LLMSettings(base_url="http://llm.test/v1", api_key="test", model="test-model")


def rag_settings() -> Settings:
    return Settings(
        embedding_base_url="http://embedding.test/v1",
        embedding_model="test-embedding",
        weaviate_host="localhost",
        weaviate_port=8080,
        weaviate_grpc_port=50051,
    )


class RetrievalPhase1Tests(unittest.TestCase):
    def test_fixed_manual_evaluation_data_has_eight_cases(self):
        cases = load_cases(Path("test/rag_manual_evaluation.json"))
        results = {
            "field_001": [{"page_number": 5, "content": "Name 病人 200"}],
            "field_002": [{"page_number": 8, "content": "RT Modality 放射 3"}],
            "rule_001": [{"page_number": 19, "content": "possible 病例不需登錄"}],
            "cross_page_001": [
                {"page_number": 16, "content": "Gastric GIST"},
                {"page_number": 17, "content": "Non-gastric GIST"},
            ],
            "unanswerable_001": [],
            "field_basic_histology_001": [{"field_code": "2.8", "content": "Histology 5 文字"}],
            "field_detail_histology_001": [
                {"field_code": "2.8", "content": "原發腫瘤細胞 收錄目的 ICD-O-3 M-code Solid Tumor coding rules"}
            ],
            "field_full_histology_001": [
                {"field_code": "2.8", "content": "Histology 欄位長度 2.8 欄位敘述 收錄目的 編碼指引"}
            ],
        }

        report = evaluate_cases(cases, results)

        self.assertEqual(len(cases), 8)
        self.assertEqual(report["metrics"], {"hit_at_5": 1.0, "hit_at_10": 1.0, "mrr": 1.0})

    def test_load_settings_reads_retrieval_limits_from_environment(self):
        environment = {
            "EMBEDDING_BASE_URL": "http://embedding.test/v1",
            "EMBEDDING_MODEL": "test-embedding",
            "WEAVIATE_HOST": "localhost",
            "WEAVIATE_PORT": "8080",
            "WEAVIATE_GRPC_PORT": "50051",
            "RAG_RETRIEVAL_TOP_K": "21",
            "RAG_RRF_CANDIDATE_TOP_K": "31",
            "RAG_RERANK_TOP_K": "9",
            "RAG_FINAL_CONTEXT_TOP_K": "7",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = load_settings()

        self.assertEqual(settings.retrieval_top_k, 21)
        self.assertEqual(settings.rrf_candidate_top_k, 31)
        self.assertEqual(settings.rerank_top_k, 9)
        self.assertEqual(settings.final_context_top_k, 7)

    def test_build_search_queries_keeps_original_and_deduplicates_rewrites(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="欄位 1.3 病人姓名\n欄位 1.3 病人姓名\nName 欄位最大長度\n表格中的 Name 代碼"))]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))

        with patch("pipeline.retrieve_answer.llm_client", return_value=client):
            queries = retrieve_answer.build_search_queries("欄位 1.3 病人姓名長度？", llm_settings())

        self.assertEqual(queries, ["欄位 1.3 病人姓名長度？", "欄位 1.3 病人姓名", "Name 欄位最大長度", "表格中的 Name 代碼"])

    def test_build_search_queries_falls_back_to_the_original_question(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(openai.OpenAIError("offline"))))
        )

        with patch("pipeline.retrieve_answer.llm_client", return_value=client):
            self.assertEqual(retrieve_answer.build_search_queries("原始問題", llm_settings()), ["原始問題"])

    def test_select_hybrid_alpha_prefers_keywords_for_structured_queries(self):
        self.assertEqual(retrieve_answer.select_hybrid_alpha("欄位 4.2.1.2 的 RT Modality 代碼 3"), 0.25)
        self.assertEqual(retrieve_answer.select_hybrid_alpha("癌症登記申報時限是多久？"), 0.65)

    def test_hybrid_retrieval_uses_configured_limits_and_alpha(self):
        item = SimpleNamespace(
            uuid="chunk-1",
            properties={"chunk_id": "chunk-1", "content": "欄位 1.3 Name 200", "page_number": 5},
            metadata=SimpleNamespace(score=0.9),
        )

        class Collection:
            def __init__(self):
                self.query = self
                self.calls = []

            def hybrid(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(objects=[item])

        collection = Collection()
        client = SimpleNamespace(close=lambda: None)
        with (
            patch("pipeline.retrieve_answer.build_search_queries", return_value=["原始問題", "改寫一", "改寫二", "改寫三"]),
            patch("pipeline.retrieve_answer.weaviate_client", return_value=client),
            patch("pipeline.retrieve_answer.rag_collection", return_value=collection),
            patch("pipeline.retrieve_answer.embedding", return_value=[0.1, 0.2]),
        ):
            chunks = retrieve_answer.retrieve_chunks("4.2.1.2 代碼 3", "pdf-1", rag_settings(), llm_settings(), "hybrid")

        self.assertEqual(len(collection.calls), 4)
        self.assertTrue(all(call["limit"] == 20 for call in collection.calls))
        self.assertTrue(all(call["alpha"] == 0.25 for call in collection.calls))
        self.assertEqual(len(chunks), 1)
        self.assertGreater(chunks[0]["rrf_score"], 0)
