import unittest
from unittest.mock import Mock, patch

from services.config import Settings
from services.vectordb import index_chunks, search_chunks


class ElasticsearchStoreTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings("http://embedding.test/v1", "embedding", "http://elasticsearch.test:9200", "documents")

    @patch("services.vectordb.requests.post")
    @patch("services.vectordb.requests.put")
    @patch("services.vectordb.embedding", return_value=[0.1, 0.2])
    def test_indexing_uses_document_scoped_ids(self, _, put, post):
        put.return_value.status_code = 201
        post.return_value.json.return_value = {"errors": False}
        post.return_value.raise_for_status = Mock()
        index_chunks("book-1", [{"chunk_id": "chunk-1", "content": "text"}], self.settings)
        payload = post.call_args.kwargs["data"]
        self.assertIn('"_id": "book-1:chunk-1"', payload)
        self.assertIn('"document_id": "book-1"', payload)

    @patch("services.vectordb.requests.put")
    def test_index_mapping_errors_are_not_silently_ignored(self, put):
        put.return_value.status_code = 400
        put.return_value.json.return_value = {"error": {"type": "mapper_parsing_exception"}}
        put.return_value.raise_for_status.side_effect = RuntimeError("mapping error")
        from services.vectordb import ensure_index
        with self.assertRaises(RuntimeError):
            ensure_index(self.settings)

    @patch("services.vectordb.requests.post")
    @patch("services.vectordb.embedding", return_value=[0.1, 0.2])
    def test_search_filters_by_document_id(self, _, post):
        post.return_value.json.return_value = {"hits": {"hits": [{"_id": "book-1:chunk-1", "_score": 1, "_source": {"document_id": "book-1", "content": "text"}}]}}
        post.return_value.raise_for_status = Mock()
        results = search_chunks("question", "book-1", self.settings, limit=5, hybrid=True)
        query = post.call_args.kwargs["json"]["query"]
        self.assertIn({"term": {"document_id": "book-1"}}, query["script_score"]["query"]["bool"]["filter"])
        self.assertEqual(results[0]["document_id"], "book-1")

    @patch("services.vectordb.requests.post")
    @patch("services.vectordb.requests.put")
    @patch("services.vectordb.embedding", return_value=[0.1, 0.2])
    def test_indexing_reports_the_rejected_chunk_reason(self, _, put, post):
        put.return_value.status_code = 400
        put.return_value.json.return_value = {"error": {"type": "resource_already_exists_exception"}}
        post.return_value.raise_for_status = Mock()
        post.return_value.json.return_value = {"errors": True, "items": [{"index": {"_id": "book-1:chunk-1", "error": {"reason": "mapper parsing failed"}}}]}

        with self.assertRaisesRegex(Exception, "chunk book-1:chunk-1: mapper parsing failed"):
            index_chunks("book-1", [{"chunk_id": "chunk-1", "content": "text"}], self.settings)

    @patch("services.vectordb.requests.post")
    @patch("services.vectordb.requests.put")
    @patch("services.vectordb.embedding", return_value=[0.1, 0.2])
    def test_indexing_reports_bulk_http_response(self, _, put, post):
        import requests
        put.return_value.status_code = 400
        put.return_value.json.return_value = {"error": {"type": "resource_already_exists_exception"}}
        response = Mock(status_code=429, text="disk watermark exceeded")
        post.return_value.raise_for_status.side_effect = requests.HTTPError(response=response)

        with self.assertRaisesRegex(Exception, "429.*disk watermark exceeded"):
            index_chunks("book-1", [{"chunk_id": "chunk-1", "content": "text"}], self.settings)

    @patch("services.vectordb.BULK_MAX_BYTES", 1)
    @patch("services.vectordb.requests.post")
    @patch("services.vectordb.requests.put")
    @patch("services.vectordb.embedding", return_value=[0.1, 0.2])
    def test_indexing_sends_large_documents_in_multiple_bulk_requests(self, _, put, post):
        put.return_value.status_code = 400
        put.return_value.json.return_value = {"error": {"type": "resource_already_exists_exception"}}
        post.return_value.raise_for_status = Mock()
        post.return_value.json.return_value = {"errors": False}

        index_chunks("book-1", [{"chunk_id": "one", "content": "first"}, {"chunk_id": "two", "content": "second"}], self.settings)

        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
