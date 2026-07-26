import unittest
from unittest.mock import Mock, patch

from pipeline.retrieve_answer import reciprocal_rank_fusion, rerank_documents, retrieve_chunks
from services.api import LLMSettings
from services.config import Settings


class ElasticsearchRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            "http://embedding.test/v1",
            "embedding",
            "http://elasticsearch.test:9200",
            "documents",
            reranker_base_url="http://reranker.test/v1",
        )
        self.llm_settings = LLMSettings("http://llm.test/v1", "test", "model")

    def test_rrf_deduplicates_chunks_and_preserves_rank(self):
        results = reciprocal_rank_fusion([
            [{"uuid": "one", "content": "one"}, {"uuid": "two", "content": "two"}],
            [{"uuid": "two", "content": "two"}],
        ])
        self.assertEqual([result["uuid"] for result in results], ["two", "one"])

    @patch("pipeline.retrieve_answer.rerank_documents", side_effect=lambda _, candidates, __: candidates)
    @patch("pipeline.retrieve_answer.search_chunks")
    @patch("pipeline.retrieve_answer.build_search_queries", return_value=["first", "second"])
    def test_retrieval_is_scoped_to_the_notebook_document(self, _, search, __):
        search.side_effect = [
            [{"uuid": "a", "document_id": "book-1", "content": "A"}],
            [{"uuid": "b", "document_id": "book-1", "content": "B"}],
        ]
        chunks = retrieve_chunks("question", "book-1", self.settings, self.llm_settings, "hybrid")
        self.assertEqual({chunk["document_id"] for chunk in chunks}, {"book-1"})
        self.assertEqual(search.call_count, 2)
        self.assertTrue(all(call.args[1] == "book-1" and call.kwargs["hybrid"] for call in search.call_args_list))

    @patch("pipeline.retrieve_answer.requests.post")
    def test_reranker_input_is_capped_below_the_model_context_limit(self, post):
        post.return_value.raise_for_status = Mock()
        post.return_value.json.return_value = {"results": [{"index": 0, "relevance_score": 1.0}]}
        rerank_documents("q" * 300, [{"content": "d" * 2000}], self.settings)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(len(payload["query"]), self.settings.reranker_max_query_chars)
        self.assertEqual(len(payload["documents"][0]), self.settings.reranker_max_document_chars)


if __name__ == "__main__":
    unittest.main()
