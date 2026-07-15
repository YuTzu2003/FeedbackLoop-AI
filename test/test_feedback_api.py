import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main


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
        self.original_feedback_log = main.app.config["FEEDBACK_LOG"]
        main.app.config.update(TESTING=True)
        main.app.config["FEEDBACK_LOG"] = Path(self.temp_dir.name) / "feedbacks.jsonl"
        self.client = main.app.test_client()

    def tearDown(self):
        main.app.config["FEEDBACK_LOG"] = self.original_feedback_log
        self.temp_dir.cleanup()

    @patch("main.weaviate.connect_to_local")
    def test_status_checks_and_closes_connection_without_data_operations(self, connect):
        fake_client = FakeWeaviateClient()
        connect.return_value = fake_client

        response = self.client.get("/api/weaviate/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"ready": True, "live": True})
        self.assertTrue(fake_client.closed)

    def test_feedback_is_appended_as_jsonl(self):
        response = self.client.post("/api/feedback", json={"score": "good", "note": "Clear", "question": "Q", "answer": "A"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["status"], "saved")
        record = json.loads(main.app.config["FEEDBACK_LOG"].read_text(encoding="utf-8"))
        self.assertEqual(record["score"], "good")
        self.assertEqual(record["note"], "Clear")

    def test_homepage_exposes_the_configured_llm_model(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(main.llm_model, response.get_data(as_text=True))

    def test_question_input_supports_enter_to_send(self):
        script = Path("static/app.js").read_text(encoding="utf-8")
        self.assertIn('event.key === "Enter" && !event.shiftKey', script)

    def test_answer_actions_use_an_accessible_copy_icon(self):
        script = Path("static/app.js").read_text(encoding="utf-8")
        self.assertIn('aria-label="\\u8907\\u88fd\\u56de\\u7b54"', script)
        self.assertIn("<svg", script)


if __name__ == "__main__":
    unittest.main()
