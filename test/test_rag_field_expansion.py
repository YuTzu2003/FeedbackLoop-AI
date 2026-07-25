import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import load_pdf, retrieve_answer
from services.api import LLMSettings
from services.config import Settings


class FieldExpansionTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [
            load_pdf.LayoutBlock(98, "title", "組織型態", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "title", "欄位長度：5", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "title", "Histology", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "title", "癌登欄位序號 #2.8", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "title", "欄位敘述：", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "text", "原發腫瘤細胞於顯微鏡下之結構。", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "title", "收錄目的：", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "text", "作為分期及決定治療方針之根據。", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "title", "編碼指引：", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(98, "text", "ICD-O-3 的 M-code 只填入數字；採用 Solid Tumor coding rules。", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(100, "title", "性態碼", (0, 0, 1, 1)),
            load_pdf.LayoutBlock(100, "title", "癌登欄位序號 #2.9", (0, 0, 1, 1)),
        ]

    def test_histology_field_chunks_include_summary_parent_and_parts(self):
        chunks = load_pdf.build_field_chunks(self.blocks)
        block_types = [chunk["block_type"] for chunk in chunks]
        summary = next(chunk for chunk in chunks if chunk["block_type"] == "field_summary")
        parent = next(chunk for chunk in chunks if chunk["block_type"] == "field_detail_parent")
        parts = [chunk for chunk in chunks if chunk["block_type"] == "field_detail_part"]

        self.assertIn("field_summary", block_types)
        self.assertIn("field_detail_parent", block_types)
        self.assertEqual({part["detail_type"] for part in parts}, {"description", "purpose", "coding_instruction"})
        self.assertEqual(summary["field_code"], "2.8")
        self.assertEqual(summary["field_name"], "組織型態")
        self.assertIn("Histology", summary["content"])
        self.assertIn("原發腫瘤細胞", parent["content"])
        self.assertTrue(all(part["parent_chunk_id"] == parent["chunk_id"] for part in parts))

    def test_histology_detail_is_not_lost_after_field_expansion(self):
        chunks = load_pdf.build_field_chunks(self.blocks)
        summary_only = [chunk for chunk in chunks if chunk["block_type"] == "field_summary"]
        expanded = retrieve_answer.order_field_chunks(summary_only, chunks, 8)
        context = "\n".join(chunk["content"] for chunk in expanded)

        self.assertNotIn("原發腫瘤細胞", summary_only[0]["content"])
        self.assertIn("原發腫瘤細胞", context)
        self.assertIn("Solid Tumor coding rules", context)

    def test_field_query_intent_distinguishes_basic_detail_and_full(self):
        self.assertEqual(retrieve_answer.classify_field_intent("組織型態欄位的英文名稱與欄位長度？"), "basic")
        self.assertEqual(retrieve_answer.classify_field_intent("組織型態欄位的收錄目的與編碼指引？"), "detail")
        self.assertEqual(retrieve_answer.classify_field_intent("請完整介紹組織型態欄位"), "full")

    def test_detail_query_uses_both_channels_and_expands_the_field(self):
        summary = SimpleNamespace(
            uuid="summary",
            properties={
                "chunk_id": "field_2_8_summary",
                "block_type": "field_summary",
                "field_code": "2.8",
                "field_name": "組織型態",
                "field_english_name": "Histology",
                "content": "Histology 5 文字",
            },
            metadata=SimpleNamespace(score=0.9),
        )
        expanded = [
            summary,
            SimpleNamespace(
                uuid="parent",
                properties={"chunk_id": "field_2_8_detail_parent", "block_type": "field_detail_parent", "field_code": "2.8", "content": "欄位敘述 收錄目的 編碼指引"},
            ),
            SimpleNamespace(
                uuid="part",
                properties={"chunk_id": "field_2_8_coding_instruction_1", "block_type": "field_detail_part", "field_code": "2.8", "detail_type": "coding_instruction", "content": "ICD-O-3 M-code Solid Tumor coding rules"},
            ),
            SimpleNamespace(
                uuid="wrong-field",
                properties={"chunk_id": "field_4_2_summary", "block_type": "field_summary", "field_code": "4.2", "field_name": "其他欄位", "content": "不應混入"},
            ),
        ]

        class Collection:
            def __init__(self):
                self.query = self
                self.calls = []
                self.fetch_calls = []

            def hybrid(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(objects=[summary])

            def fetch_objects(self, **kwargs):
                self.fetch_calls.append(kwargs)
                return SimpleNamespace(objects=expanded)

        collection = Collection()
        client = SimpleNamespace(close=lambda: None)
        settings = Settings("http://embedding.test/v1", "embedding", "localhost", 8080, 50051)
        llm_settings = LLMSettings("http://llm.test/v1", "test", "model")
        with (
            patch("pipeline.retrieve_answer.build_search_queries", return_value=["組織型態欄位的收錄目的與編碼指引？"]),
            patch("pipeline.retrieve_answer.weaviate_client", return_value=client),
            patch("pipeline.retrieve_answer.rag_collection", return_value=collection),
            patch("pipeline.retrieve_answer.embedding", return_value=[0.1]),
        ):
            chunks = retrieve_answer.retrieve_chunks("組織型態欄位的收錄目的與編碼指引？", "pdf-1", settings, llm_settings, "hybrid")

        self.assertTrue(collection.fetch_calls)
        self.assertIn("組織型態 收錄目的 編碼指引", [call["query"] for call in collection.calls])
        self.assertEqual([chunk["block_type"] for chunk in chunks[:3]], ["field_summary", "field_detail_parent", "field_detail_part"])
        self.assertEqual({chunk["field_code"] for chunk in chunks}, {"2.8"})
