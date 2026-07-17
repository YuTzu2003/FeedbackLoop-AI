import unittest
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeWeaviateClient:
    closed = False

    def is_ready(self):
        return True

    def is_live(self):
        return True

    def close(self):
        self.closed = True


class WeaviateConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.original_feedback_log = app.app.config["FEEDBACK_LOG"]
        self.original_notebook_log = app.app.config["NOTEBOOK_LOG"]
        self.original_notebook_history_log = app.app.config["NOTEBOOK_HISTORY_LOG"]
        self.original_upload_folder = app.app.config["UPLOAD_FOLDER"]
        app.app.config.update(TESTING=True)
        app.app.config["FEEDBACK_LOG"] = Path(self.temp_dir.name) / "feedbacks.jsonl"
        app.app.config["NOTEBOOK_LOG"] = Path(self.temp_dir.name) / "notebooks.jsonl"
        app.app.config["NOTEBOOK_HISTORY_LOG"] = Path(self.temp_dir.name) / "notebook_history.jsonl"
        app.app.config["UPLOAD_FOLDER"] = Path(self.temp_dir.name) / "uploads"
        self.client = app.app.test_client()

    def tearDown(self):
        app.app.config["FEEDBACK_LOG"] = self.original_feedback_log
        app.app.config["NOTEBOOK_LOG"] = self.original_notebook_log
        app.app.config["NOTEBOOK_HISTORY_LOG"] = self.original_notebook_history_log
        app.app.config["UPLOAD_FOLDER"] = self.original_upload_folder
        self.temp_dir.cleanup()

    @patch("app.weaviate.connect_to_local")
    def test_status_checks_and_closes_connection_without_data_operations(self, connect):
        fake_client = FakeWeaviateClient()
        connect.return_value = fake_client

        response = self.client.get("/api/weaviate/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"ready": True, "live": True})
        self.assertTrue(fake_client.closed)

    def test_feedback_is_appended_as_jsonl(self):
        payload = {"score": "good", "note": "Clear", "question": "Q", "answer": "A", "history_id": "answer-1"}
        response = self.client.post("/api/feedback", json=payload)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["status"], "saved")
        record = json.loads(app.app.config["FEEDBACK_LOG"].read_text(encoding="utf-8"))
        self.assertEqual(record["score"], "good")
        self.assertEqual(record["note"], "Clear")
        self.assertEqual(record["history_id"], "answer-1")
        listed = self.client.get("/api/feedbacks")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json["items"][0]["question"], "Q")
        duplicate = self.client.post("/api/feedback", json=payload)
        self.assertEqual(duplicate.status_code, 409)

    def test_negative_feedback_requires_a_note(self):
        response = self.client.post("/api/feedback", json={"score": "bad", "note": " ", "history_id": "answer-2"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("請說明需要改善的地方", response.json["error"])

    def test_feedback_and_connection_pages_use_the_correct_templates(self):
        self.assertIn("回饋紀錄", self.client.get("/feedback").get_data(as_text=True))
        self.assertIn("Weaviate", self.client.get("/connection").get_data(as_text=True))
        self.assertEqual(self.client.get("/feedbacks").status_code, 200)

    def test_feedback_page_restores_metrics_and_filters(self):
        page = self.client.get("/feedback").get_data(as_text=True)
        self.assertIn('id="total"', page)
        self.assertIn('data-filter="good"', page)

    def test_upload_creates_a_notebook(self):
        response = self.client.post("/api/upload", data={"file": (BytesIO(b"name,value\na,1\n"), "report.csv")})

        self.assertEqual(response.status_code, 200)
        notebook = app.read_jsonl(app.app.config["NOTEBOOK_LOG"], "notebook")[0]
        self.assertEqual(notebook["id"], response.json["notebook_id"])
        self.assertEqual(notebook["name"], "report.csv")
        self.assertTrue((app.app.config["UPLOAD_FOLDER"] / notebook["stored_filename"]).exists())

    @patch.object(app.llm_client.chat.completions, "create")
    def test_notebook_history_is_sent_to_llm_and_new_answer_is_saved(self, create):
        notebook = {"id": "book-1", "name": "report.csv", "created_at": "2026-01-01T00:00:00+00:00"}
        old_record = {"id": "old-1", "notebook_id": "book-1", "question": "What was the revenue?", "answer": "100", "created_at": "2026-01-01T00:00:00+00:00"}
        app.app.config["NOTEBOOK_LOG"].write_text(json.dumps(notebook) + "\n", encoding="utf-8")
        app.app.config["NOTEBOOK_HISTORY_LOG"].write_text(json.dumps(old_record) + "\n", encoding="utf-8")
        create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="The revenue increased."))])

        response = self.client.post("/api/ask", json={"question": "Compare it with today.", "notebook_id": "book-1"})

        self.assertEqual(response.status_code, 200)
        messages = create.call_args.kwargs["messages"]
        self.assertIn({"role": "user", "content": "What was the revenue?"}, messages)
        self.assertEqual(len(app.notebook_history("book-1")), 2)

    def test_notebook_apis_scope_history_to_the_requested_notebook(self):
        app.app.config["NOTEBOOK_LOG"].write_text('{"id":"book-1","name":"report.csv","created_at":"2026-01-01T00:00:00+00:00"}\n', encoding="utf-8")
        app.app.config["NOTEBOOK_HISTORY_LOG"].write_text('{"id":"1","notebook_id":"book-1","question":"Q","answer":"A","created_at":"2026-01-01T00:00:00+00:00"}\n{"id":"2","notebook_id":"book-2","question":"Other","answer":"B","created_at":"2026-01-01T00:00:00+00:00"}\n', encoding="utf-8")
        response = self.client.get("/api/notebooks/book-1/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["items"][0]["id"], "1")
        self.assertEqual(self.client.get("/api/notebooks").json["items"][0]["id"], "book-1")

    def test_question_requires_an_existing_notebook(self):
        response = self.client.post("/api/ask", json={"question": "Q"})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/ask", json={"question": "Q", "notebook_id": "missing"})
        self.assertEqual(response.status_code, 404)

    def test_homepage_exposes_the_configured_llm_model(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(app.llm_model, response.get_data(as_text=True))

    def test_homepage_inherits_base_template_and_marks_its_navigation_active(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("文件問答｜FeedbackLoop AI", page)
        self.assertIn('<a class="active" href="/">資料問答</a>', page)
        self.assertIn("bootstrap-icons", page)

    def test_question_input_supports_enter_to_send(self):
        script = (PROJECT_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('event.key === "Enter" && !event.shiftKey', script)

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
