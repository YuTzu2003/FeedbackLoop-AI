import unittest
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace

import app
from services.api import load_llm_settings
from services.notebook_repositories import _notebook_from_row
from services.notebook_repositories import notebook_data_dir, notebook_history_path
from pipeline.load_pdf import write_pdf_chunk_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WeaviateConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.original_notebook_data_root = app.app.config["NOTEBOOK_DATA_ROOT"]
        self.original_pdf_chunk_report_dir = app.app.config["PDF_CHUNK_REPORT_DIR"]
        app.app.config.update(TESTING=True)
        app.app.config["NOTEBOOK_DATA_ROOT"] = Path(self.temp_dir.name) / "notebooks"
        app.app.config["PDF_CHUNK_REPORT_DIR"] = Path(self.temp_dir.name) / "pdf_chunks"
        self.client = app.app.test_client()
        with self.client.session_transaction() as client_session:
            client_session["id"] = "test-id"
            client_session["position"] = "Admin"
        self.notebooks = {}
        self.notebook_patches = [
            patch("app.get_notebook", side_effect=self.get_notebook),
            patch("app.list_notebook_records", side_effect=self.list_notebooks),
            patch("app.create_notebook", side_effect=self.create_notebook),
            patch("app.delete_notebook_record", side_effect=self.delete_notebook),
        ]
        for notebook_patch in self.notebook_patches:
            notebook_patch.start()

    def tearDown(self):
        app.app.config["NOTEBOOK_DATA_ROOT"] = self.original_notebook_data_root
        app.app.config["PDF_CHUNK_REPORT_DIR"] = self.original_pdf_chunk_report_dir
        for notebook_patch in self.notebook_patches:
            notebook_patch.stop()
        self.temp_dir.cleanup()

    def get_notebook(self, notebook_id, owner_user_id):
        notebook = self.notebooks.get(notebook_id)
        return notebook if notebook and notebook["owner_user_id"] == str(owner_user_id) else None

    def list_notebooks(self, owner_user_id):
        return [notebook for notebook in self.notebooks.values() if notebook["owner_user_id"] == str(owner_user_id)]

    def create_notebook(self, owner_user_id, notebook):
        self.notebooks[notebook["id"]] = {**notebook, "owner_user_id": str(owner_user_id)}

    def delete_notebook(self, notebook_id, owner_user_id):
        if self.get_notebook(notebook_id, owner_user_id):
            del self.notebooks[notebook_id]

    @patch("app.weaviate_status", return_value={"ready": True, "live": True})
    def test_status_uses_the_rag_service(self, status):

        response = self.client.get("/api/weaviate/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"ready": True, "live": True})
        status.assert_called_once_with(app.settings)

    @patch("services.feedback.create_feedback", return_value="feedback-1")
    @patch("services.feedback.find_user_history", return_value={"id": "answer-1", "question": "Q", "answer": "A"})
    def test_feedback_is_saved_to_mssql(self, find_history, create_feedback):
        payload = {"score": "good", "note": "Clear", "question": "forged", "answer": "forged", "history_id": "answer-1"}
        response = self.client.post("/api/feedback", json=payload)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["status"], "saved")
        self.assertEqual(response.json["feedback_id"], "feedback-1")
        find_history.assert_called_once_with(app.app.config["NOTEBOOK_DATA_ROOT"], "test-id", "answer-1")
        create_feedback.assert_called_once_with("test-id", {"id": "answer-1", "question": "Q", "answer": "A"}, "good", "Clear")

    def test_negative_feedback_requires_a_note(self):
        response = self.client.post("/api/feedback", json={"score": "bad", "note": " ", "history_id": "answer-1"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("note is required", response.json["error"])

    def test_feedback_requires_an_existing_answer(self):
        response = self.client.post("/api/feedback", json={"score": "good", "note": "Clear", "history_id": "missing"})
        self.assertEqual(response.status_code, 404)

    def test_connection_page_uses_the_correct_template(self):
        self.assertIn("Weaviate", self.client.get("/connection").get_data(as_text=True))

    @patch("services.feedback.list_feedback", return_value=[{"score": "good", "question": "Q", "answer": "A", "note": "", "created_at": "2026-01-01T00:00:00+00:00"}])
    def test_feedback_page_and_api_use_mssql_records(self, list_feedback):
        page = self.client.get("/feedback").get_data(as_text=True)
        response = self.client.get("/api/feedbacks")

        self.assertIn("回饋紀錄", page)
        self.assertIn('js/feedback.js', page)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["items"][0]["score"], "good")
        list_feedback.assert_called_once_with("test-id")

    def test_upload_creates_a_notebook(self):
        response = self.client.post("/api/upload", data={"file": (BytesIO(b"name,value\na,1\n"), "report.csv")})

        self.assertEqual(response.status_code, 200)
        notebook = self.notebooks[response.json["notebook_id"]]
        self.assertEqual(notebook["id"], response.json["notebook_id"])
        self.assertEqual(notebook["name"], "report.csv")
        self.assertEqual(notebook["owner_user_id"], "test-id")
        notebook_dir = notebook_data_dir(app.app.config["NOTEBOOK_DATA_ROOT"], "test-id", notebook["id"])
        self.assertTrue((notebook_dir / notebook["stored_filename"]).exists())

    def test_sql_notebook_uses_the_new_column_names(self):
        notebook = _notebook_from_row(
            SimpleNamespace(
                NotebookId="book-1",
                UserID=7,
                Title="Quarterly report",
                StoreFilename="book-1_report.csv",
                SourceType="file",
                Url=None,
                ChunkCount=None,
                CreatedAt=datetime(2026, 1, 1, 9, 0),
            )
        )

        self.assertEqual(notebook["name"], "Quarterly report")
        self.assertEqual(notebook["stored_filename"], "book-1_report.csv")
        self.assertEqual(notebook["owner_user_id"], "7")

    @patch("app.ingest_pdf", return_value={"source_type": "pdf", "chunk_count": 2, "processed_pages": 1, "ocr_pages": 0})
    def test_pdf_upload_indexes_chunks_before_creating_notebook(self, ingest):
        response = self.client.post("/api/upload", data={"file": (BytesIO(b"%PDF-1.4"), "report.pdf")})

        self.assertEqual(response.status_code, 200)
        notebook = self.notebooks[response.json["notebook_id"]]
        self.assertEqual(notebook["source_type"], "pdf")
        self.assertEqual(response.json["chunk_count"], 2)
        ingest.assert_called_once_with(
            notebook_data_dir(app.app.config["NOTEBOOK_DATA_ROOT"], "test-id", notebook["id"]) / notebook["stored_filename"],
            document_id=notebook["id"],
            filename="report.pdf",
            settings=app.settings,
            report_dir=app.app.config["PDF_CHUNK_REPORT_DIR"],
        )

    def test_pdf_chunk_report_is_saved_as_inspectable_json(self):
        report = {"source": "report.pdf", "chunks": [{"chunk_id": "chunk_00001", "content": "Example text"}]}
        report_path = write_pdf_chunk_report(report, Path(self.temp_dir.name) / "pdf_chunks", "pdf-1")

        self.assertEqual(report_path.name, "pdf-1.json")
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)

    @patch("app.ingest_web_url", return_value={"id": "web-1", "name": "Example page", "source_type": "web", "url": "https://example.com", "chunk_count": 2})
    def test_url_upload_creates_a_web_notebook_after_chunking(self, ingest):
        response = self.client.post("/api/upload_url", json={"url": "https://example.com"})

        self.assertEqual(response.status_code, 200)
        notebook = self.notebooks[response.json["notebook_id"]]
        self.assertEqual(notebook["source_type"], "web")
        self.assertEqual(notebook["url"], "https://example.com")
        self.assertEqual(notebook["chunk_count"], 2)
        ingest.assert_called_once_with("https://example.com", app.settings)

    def test_url_upload_rejects_an_empty_url(self):
        response = self.client.post("/api/upload_url", json={"url": ""})

        self.assertEqual(response.status_code, 400)

    @patch("app.answer_from_chunks", return_value="答案來自來源內容")
    @patch("app.retrieve_chunks", return_value=[{"title": "Example page", "url": "https://example.com", "chunk_index": 1, "score": 0.91, "content": "source text"}])
    def test_web_notebook_question_uses_document_scoped_retrieval(self, retrieve, answer):
        self.notebooks["web-1"] = {"id": "web-1", "name": "Example page", "source_type": "web", "owner_user_id": "test-id", "created_at": "2026-01-01T00:00:00+00:00"}

        response = self.client.post("/api/ask", json={"question": "What is the source?", "notebook_id": "web-1"})

        self.assertEqual(response.status_code, 200)
        retrieve.assert_called_once_with("What is the source?", "web-1", app.settings, app.llm_settings, "near_vector")
        answer.assert_called_once_with("What is the source?", retrieve.return_value, app.llm_settings)
        self.assertEqual(response.json["sources"][0]["url"], "https://example.com")

    @patch("app.answer_from_chunks", return_value="PDF answer")
    @patch("app.retrieve_chunks", return_value=[{"source_type": "pdf", "title": "report.pdf", "page_number": 2, "chunk_index": 1, "score": 0.91, "content": "source text"}])
    def test_pdf_notebook_question_uses_document_scoped_retrieval(self, retrieve, answer):
        self.notebooks["pdf-1"] = {"id": "pdf-1", "name": "report.pdf", "source_type": "pdf", "owner_user_id": "test-id", "created_at": "2026-01-01T00:00:00+00:00"}

        response = self.client.post("/api/ask", json={"question": "What is the source?", "notebook_id": "pdf-1", "search_mode": "hybrid"})

        self.assertEqual(response.status_code, 200)
        retrieve.assert_called_once_with("What is the source?", "pdf-1", app.settings, app.llm_settings, "hybrid")
        answer.assert_called_once_with("What is the source?", retrieve.return_value, app.llm_settings)
        self.assertEqual(response.json["sources"][0]["page_number"], 2)

    @patch("app.answer_from_history", return_value="The revenue increased.")
    def test_notebook_history_is_sent_to_llm_and_new_answer_is_saved(self, answer):
        notebook = {"id": "book-1", "name": "report.csv", "owner_user_id": "test-id", "created_at": "2026-01-01T00:00:00+00:00"}
        old_record = {"id": "old-1", "notebook_id": "book-1", "question": "What was the revenue?", "answer": "100", "created_at": "2026-01-01T00:00:00+00:00"}
        self.notebooks["book-1"] = notebook
        history_path = notebook_history_path(app.app.config["NOTEBOOK_DATA_ROOT"], "test-id", "book-1")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(old_record) + "\n", encoding="utf-8")
        response = self.client.post("/api/ask", json={"question": "Compare it with today.", "notebook_id": "book-1"})

        self.assertEqual(response.status_code, 200)
        messages = answer.call_args.args[0]
        self.assertIn({"role": "user", "content": "What was the revenue?"}, messages)
        self.assertEqual(len(app.notebook_history(history_path, "book-1")), 2)

    def test_notebook_apis_scope_history_to_the_requested_notebook(self):
        self.notebooks["book-1"] = {"id": "book-1", "name": "report.csv", "owner_user_id": "test-id", "created_at": "2026-01-01T00:00:00+00:00"}
        history_path = notebook_history_path(app.app.config["NOTEBOOK_DATA_ROOT"], "test-id", "book-1")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text('{"id":"1","notebook_id":"book-1","question":"Q","answer":"A","created_at":"2026-01-01T00:00:00+00:00"}\n', encoding="utf-8")
        response = self.client.get("/api/notebooks/book-1/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["items"][0]["id"], "1")
        self.assertEqual(self.client.get("/api/notebooks").json["items"][0]["id"], "book-1")

    def test_notebooks_are_isolated_by_logged_in_user(self):
        self.notebooks["book-a"] = {"id": "book-a", "name": "A", "owner_user_id": "user-a"}
        self.notebooks["book-b"] = {"id": "book-b", "name": "B", "owner_user_id": "user-b"}
        with self.client.session_transaction() as client_session:
            client_session["id"] = "user-a"
            client_session["position"] = "Admin"

        self.assertEqual(self.client.get("/api/notebooks").json["items"], [{"id": "book-a", "name": "A", "owner_user_id": "user-a"}])
        self.assertEqual(self.client.get("/api/notebooks/book-b/history").status_code, 404)
        self.assertEqual(self.client.post("/api/ask", json={"question": "Q", "notebook_id": "book-b"}).status_code, 404)
        self.assertEqual(self.client.delete("/api/notebooks/book-b").status_code, 404)

    @patch("app.delete_document")
    def test_deleting_a_notebook_removes_its_entire_data_directory(self, delete_document):
        self.notebooks["book-1"] = {"id": "book-1", "name": "report.csv", "stored_filename": "report.csv", "owner_user_id": "test-id"}
        notebook_dir = notebook_data_dir(app.app.config["NOTEBOOK_DATA_ROOT"], "test-id", "book-1")
        notebook_dir.mkdir(parents=True)
        (notebook_dir / "report.csv").write_text("name,value\na,1\n", encoding="utf-8")
        (notebook_dir / "history.jsonl").write_text('{"id":"answer-1"}\n', encoding="utf-8")

        response = self.client.delete("/api/notebooks/book-1")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("book-1", self.notebooks)
        self.assertFalse(notebook_dir.exists())
        delete_document.assert_called_once_with("book-1", app.settings)

    def test_question_requires_an_existing_notebook(self):
        response = self.client.post("/api/ask", json={"question": "Q"})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/ask", json={"question": "Q", "notebook_id": "missing"})
        self.assertEqual(response.status_code, 404)

    def test_question_rejects_an_invalid_search_mode(self):
        response = self.client.post("/api/ask", json={"question": "Q", "notebook_id": "book-1", "search_mode": "invalid"})

        self.assertEqual(response.status_code, 400)

    def test_homepage_exposes_the_configured_llm_model(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(app.llm_settings.model, response.get_data(as_text=True))

    def test_model_settings_require_environment_values(self):
        with patch.dict(os.environ, {"LLM_MODEL": ""}):
            with self.assertRaisesRegex(RuntimeError, "LLM_MODEL"):
                load_llm_settings()

    def test_homepage_inherits_base_template_and_marks_its_navigation_active(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("文件問答｜FeedbackLoop AI", page)
        self.assertIn('<a class="active" href="/">資料問答</a>', page)
        self.assertIn("bootstrap-icons", page)

    def test_question_input_supports_enter_to_send(self):
        script = (PROJECT_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('event.key === "Enter" && !event.shiftKey', script)
        self.assertIn('search_mode: searchMode.value', script)

    def test_homepage_defaults_to_near_vector_search(self):
        page = self.client.get("/").get_data(as_text=True)

        self.assertIn('id="searchMode"', page)
        self.assertIn('value="near_vector" selected', page)

    def test_answer_actions_use_an_accessible_copy_icon(self):
        script = (PROJECT_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
        styles = (PROJECT_ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
        self.assertIn('aria-label="複製回答"', script)
        self.assertIn('bi bi-copy', script)
        self.assertIn('bi bi-check2', script)
        self.assertNotIn('ai-answer-icon', script)
        self.assertIn('bi bi-hand-thumbs-up', script)
        self.assertIn('bi bi-hand-thumbs-down', script)
        self.assertIn('class="answer-actions"', script)
        self.assertLess(script.index('data-score="good"'), script.index('data-score="bad"'))
        self.assertLess(script.index('data-score="bad"'), script.index('data-copy'))
        self.assertNotIn('這個回答有幫助嗎', script)
        self.assertIn('scoreButton.dataset.score === "good"', script)
        self.assertIn('DOMPurify.sanitize(marked.parse', script)
        self.assertIn('history_id: feedback.dataset.historyId', script)
        self.assertIn('.answer-actions', styles)
        self.assertIn('line-height: 1', styles)

    def test_homepage_loads_bootstrap_icons(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("bootstrap-icons", page)
        self.assertIn("marked/marked.min.js", page)
        self.assertIn("dompurify/dist/purify.min.js", page)


if __name__ == "__main__":
    unittest.main()
