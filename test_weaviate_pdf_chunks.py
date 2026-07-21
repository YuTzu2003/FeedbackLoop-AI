import json

from weaviate_pdf_chunks import embed_texts, load_chunks


def test_load_chunks_maps_pdf_report_to_weaviate_properties(tmp_path) -> None:
    report_path = tmp_path / "ragflow_pdf_test.json"
    report_path.write_text(
        json.dumps(
            {
                "source": "data/manual.pdf",
                "total_pages": 2,
                "chunks": [
                    {
                        "chunk_id": "chunk_00001",
                        "page_number": 2,
                        "layout_types": ["text"],
                        "content": "Cancer registry content",
                        "source_blocks": [{"bbox": [1, 2, 3, 4]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    chunks = load_chunks(report_path, document_id="doc_pdf_001", user_id="user_001")

    assert chunks == [
        {
            "chunk_id": "chunk_00001",
            "document_id": "doc_pdf_001",
            "user_id": "user_001",
            "source_type": "pdf",
            "content": "Cancer registry content",
            "filename": "manual.pdf",
            "page_number": 2,
            "sheet_name": "",
            "row_start": 0,
            "row_end": 0,
            "url": "",
            "title": "",
            "chunk_index": 1,
            "layout_types": ["text"],
            "source_blocks_json": '[{"bbox": [1, 2, 3, 4]}]',
            "created_at": chunks[0]["created_at"],
        }
    ]


def test_embed_texts_uses_ollama_batch_endpoint(monkeypatch) -> None:
    calls = []

    class Response:
        ok = True

        def json(self):
            return {"embeddings": [[1.0], [2.0]]}

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setattr("services.weaviate_pdf_chunks.requests.post", fake_post)

    embeddings = embed_texts(["a", "b"], base_url="http://ollama", model="embed-model", timeout=3)

    assert embeddings == [[1.0], [2.0]]
    assert calls == [("http://ollama/api/embed", {"model": "embed-model", "input": ["a", "b"]}, 3)]


def test_embed_texts_falls_back_to_legacy_ollama_endpoint(monkeypatch) -> None:
    calls = []

    class BatchResponse:
        ok = False

    class FallbackResponse:
        ok = True

        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": [self.value]}

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        if url.endswith("/api/embed"):
            return BatchResponse()
        return FallbackResponse(len(calls))

    monkeypatch.setattr("services.weaviate_pdf_chunks.requests.post", fake_post)

    embeddings = embed_texts(["a", "b"], base_url="http://ollama/", model="embed-model")

    assert embeddings == [[2], [3]]
    assert calls[1][0] == "http://ollama/api/embeddings"
    assert calls[1][1] == {"model": "embed-model", "prompt": "a"}
