import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.load_spreadsheet import merge_sections, spreadsheet_sections


class SpreadsheetIngestionTests(unittest.TestCase):
    def test_csv_rows_keep_headers_and_are_chunked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            path.write_text("name,score\nAda,100\n", encoding="utf-8")
            sections = spreadsheet_sections(path)
        self.assertEqual(sections, ["name: Ada; score: 100; Sheet: Sheet1"])
        self.assertEqual(merge_sections(sections), sections)

    def test_xlsx_rows_keep_their_sheet_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({"item": ["Widget"], "quantity": [3]}).to_excel(writer, sheet_name="Inventory", index=False)
            sections = spreadsheet_sections(path)
        self.assertEqual(sections, ["item: Widget; quantity: 3; Sheet: Inventory"])
