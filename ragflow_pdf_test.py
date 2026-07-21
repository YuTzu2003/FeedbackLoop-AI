"""A small, runnable PDF test pipeline modelled on RAGFlow DeepDoc.

The production RAGFlow parser also runs ONNX layout and table-structure
models.  This module keeps the same useful output contract (page decision,
layout blocks, provenance-aware chunks) without vendoring RAGFlow or its model
bundle.  It uses PyMuPDF's optional Tesseract bridge only for pages whose text
layer is absent or garbled.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz


_CID = re.compile(r"\(cid\s*:\s*\d+\s*\)", re.IGNORECASE)


@dataclass(frozen=True)
class LayoutBlock:
    page: int
    kind: str
    text: str
    bbox: tuple[float, float, float, float]


def garbled_ratio(text: str) -> float:
    """Return the fraction of visible characters unusable for retrieval."""
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 1.0
    garbled = sum(
        char == "\ufffd"
        or 0xE000 <= ord(char) <= 0xF8FF
        or 0x80 <= ord(char) <= 0x9F
        for char in visible
    )
    if _CID.search(text):
        garbled = max(garbled, 1)
    return garbled / len(visible)


def needs_ocr(text: str, threshold: float = 0.5) -> bool:
    """Mirror RAGFlow's important rule: garbled text is not usable text."""
    return not text.strip() or bool(_CID.search(text)) or garbled_ratio(text) >= threshold


def has_visible_ink(page: fitz.Page) -> bool:
    """Distinguish a scanned page from an intentionally blank PDF page."""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(0.25, 0.25), colorspace=fitz.csGRAY, alpha=False)
    return sum(value < 245 for value in pixmap.samples) > 25


def classify_blocks(page_number: int, page_height: float, raw_blocks: Iterable[tuple]) -> list[LayoutBlock]:
    """Provide a deterministic layout fallback when DeepDoc's ONNX model is absent."""
    blocks: list[LayoutBlock] = []
    for block in raw_blocks:
        x0, top, x1, bottom, text, _, block_type = block[:7]
        line_count = text.count("\n") + 1
        text = " ".join(text.split())
        if block_type != 0 or not text:
            continue
        if top < page_height * 0.06:
            kind = "header"
        elif bottom > page_height * 0.94:
            kind = "footer"
        elif top < page_height * 0.35 and line_count <= 2 and len(text) <= 80:
            kind = "title"
        else:
            kind = "text"
        blocks.append(LayoutBlock(page_number, kind, text, (x0, top, x1, bottom)))
    return sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0]))


def build_chunks(blocks: Iterable[LayoutBlock], chunk_size: int, overlap: int) -> list[dict]:
    """Join adjacent readable blocks while retaining every source rectangle."""
    chunks: list[dict] = []
    current: list[LayoutBlock] = []
    current_length = 0

    def flush(keep_overlap: bool) -> None:
        nonlocal current, current_length
        if not current:
            return
        text = "\n\n".join(block.text for block in current)
        chunks.append(
            {
                "chunk_id": f"chunk_{len(chunks) + 1:05d}",
                "page_number": current[0].page,
                "layout_types": sorted({block.kind for block in current}),
                "content": text,
                "source_blocks": [asdict(block) for block in current],
            }
        )
        if keep_overlap and overlap and text:
            tail = text[-overlap:]
            last = current[-1]
            current = [LayoutBlock(last.page, "overlap", tail, last.bbox)]
            current_length = len(tail)
        else:
            current = []
            current_length = 0

    for block in blocks:
        if block.kind in {"header", "footer"}:
            continue
        extra = len(block.text) + (2 if current else 0)
        if current and block.page != current[0].page:
            flush(keep_overlap=False)
        if current and current_length + extra > chunk_size:
            flush(keep_overlap=True)
        current.append(block)
        current_length += extra
    flush(keep_overlap=False)
    return chunks


def parse_pdf(
    pdf_path: Path, *, max_pages: int | None, use_ocr: bool, force_ocr: bool, language: str, chunk_size: int, overlap: int
) -> dict:
    document = fitz.open(pdf_path)
    page_limit = min(len(document), max_pages) if max_pages else len(document)
    layout_blocks: list[LayoutBlock] = []
    page_decisions: list[dict] = []

    for index in range(page_limit):
        page = document[index]
        native_text = page.get_text("text")
        required = force_ocr or (needs_ocr(native_text) and (bool(native_text.strip()) or has_visible_ink(page)))
        textpage = None
        ocr_error = None
        if required and use_ocr:
            try:
                textpage = page.get_textpage_ocr(language=language, dpi=300, full=True)
            except RuntimeError as error:
                ocr_error = str(error)
        raw_blocks = page.get_text("blocks", textpage=textpage)
        layout_blocks.extend(classify_blocks(index + 1, page.rect.height, raw_blocks))
        page_decisions.append(
            {
                "page": index + 1,
                "native_characters": len(native_text.strip()),
                "garbled_ratio": round(garbled_ratio(native_text), 4),
                "ocr_required": required,
                "ocr_used": textpage is not None,
                "ocr_error": ocr_error,
            }
        )

    return {
        "source": str(pdf_path),
        "total_pages": len(document),
        "processed_pages": page_limit,
        "ocr_pages": sum(item["ocr_used"] for item in page_decisions),
        "ocr_required_pages": sum(item["ocr_required"] for item in page_decisions),
        "page_decisions": page_decisions,
        "layout_blocks": [asdict(block) for block in layout_blocks],
        "chunks": build_chunks(layout_blocks, chunk_size, overlap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a RAGFlow-inspired PDF layout and chunk pipeline.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf/ragflow_pdf_test.json"))
    parser.add_argument("--max-pages", type=int, help="Use a small page range while validating OCR.")
    parser.add_argument("--ocr", action="store_true", help="Run Tesseract OCR only for pages that need it.")
    parser.add_argument("--force-ocr", action="store_true", help="OCR every processed page; implies --ocr.")
    parser.add_argument("--ocr-language", default="chi_tra+eng")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF does not exist: {args.pdf}")
    if args.chunk_size <= 0 or args.chunk_overlap < 0:
        parser.error("chunk sizes must be non-negative, and chunk-size must be positive")

    result = parse_pdf(
        args.pdf,
        max_pages=args.max_pages,
        use_ocr=args.ocr or args.force_ocr,
        force_ocr=args.force_ocr,
        language=args.ocr_language,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Processed {result['processed_pages']}/{result['total_pages']} pages; "
        f"OCR required: {result['ocr_required_pages']}; OCR used: {result['ocr_pages']}; "
        f"chunks: {len(result['chunks'])}; report: {args.output}"
    )


if __name__ == "__main__":
    main()
