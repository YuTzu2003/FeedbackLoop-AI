from ragflow_pdf_test import LayoutBlock, build_chunks, garbled_ratio, needs_ocr


def test_garbled_text_requires_ocr() -> None:
    assert needs_ocr("\ufffd\ufffd\ufffd")
    assert garbled_ratio("正常中文") == 0


def test_chunks_keep_page_and_source_boxes() -> None:
    blocks = [
        LayoutBlock(1, "title", "第一章", (1, 1, 10, 10)),
        LayoutBlock(1, "text", "內容", (1, 11, 10, 20)),
        LayoutBlock(2, "text", "下一頁", (1, 1, 10, 10)),
    ]
    chunks = build_chunks(blocks, chunk_size=100, overlap=10)
    assert [chunk["page_number"] for chunk in chunks] == [1, 2]
    assert chunks[0]["source_blocks"][0]["bbox"] == (1, 1, 10, 10)
