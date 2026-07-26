import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.load_pdf import ragflow_naive_merge


class RAGFlowPlainPdfTests(unittest.TestCase):
    def test_plain_sections_are_merged_in_reading_order(self):
        chunks = ragflow_naive_merge([
            {"page_number": 1, "content": "one two"},
            {"page_number": 1, "content": "three four"},
            {"page_number": 2, "content": "five six"},
        ], chunk_token_num=4)
        self.assertEqual([chunk["content"] for chunk in chunks], ["one two\nthree four", "five six"])
        self.assertEqual(chunks[1]["page_start"], 2)
