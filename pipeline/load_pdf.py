from __future__ import annotations
import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import fitz
from services.config import Settings
from services.vectordb import (CHUNK_OVERLAP,CHUNK_SIZE,RagServiceError,embedding,rag_collection,weaviate_client,)

_CID = re.compile(r"\(cid\s*:\s*\d+\s*\)", re.IGNORECASE)
FIELD_CODE_PATTERN = re.compile(r"癌登欄位序號\s*#?\s*([0-9]+(?:\.[0-9]+)*)")
FIELD_LENGTH_PATTERN = re.compile(r"欄位長度[：:]\s*(\d+)")
SUMMARY_ROW_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)+)\s+(.+?)\s+([A-Za-z][A-Za-z /-]+?)\s+(\d+)\s+(文字|數字|日期|英數)\s*$"
)
DETAIL_HEADINGS = (
    ("欄位敘述", "description"),
    ("收錄目的", "purpose"),
    ("編碼指引", "coding_instruction"),
    ("附註", "note"),
    ("注意", "note"),
    ("例外", "exception"),
)

@dataclass(frozen=True)
class LayoutBlock:
    page: int
    kind: str
    text: str
    bbox: tuple[float, float, float, float]

def garbled_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 1.0
    garbled = sum(char == "\ufffd" or 0xE000 <= ord(char) <= 0xF8FF or 0x80 <= ord(char) <= 0x9F
        for char in visible)
    
    if _CID.search(text):
        garbled = max(garbled, 1)
    return garbled / len(visible)

def needs_ocr(text: str, threshold: float = 0.5) -> bool:
    return not text.strip() or bool(_CID.search(text)) or garbled_ratio(text) >= threshold

def has_visible_ink(page: fitz.Page) -> bool:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(0.25, 0.25), colorspace=fitz.csGRAY, alpha=False)
    return sum(value < 245 for value in pixmap.samples) > 25

def classify_blocks(page_number: int, page_height: float, raw_blocks: Iterable[tuple]) -> list[LayoutBlock]:
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


def detail_type_for(text: str) -> str | None:
    normalized = text.strip().lstrip("•")
    for marker, detail_type in DETAIL_HEADINGS:
        if normalized.startswith(marker):
            return detail_type
    return None


def field_header_start(blocks: list[LayoutBlock], code_index: int) -> int:
    start = code_index
    page = blocks[code_index].page
    while start and blocks[start - 1].page == page and blocks[start - 1].kind == "title":
        start -= 1
    return start


def field_header_values(header_blocks: list[LayoutBlock], code: str) -> dict:
    texts = [block.text for block in header_blocks]
    length_match = FIELD_LENGTH_PATTERN.search(" ".join(texts))
    english_name = next(
        (
            text
            for text in texts
            if re.fullmatch(r"[A-Za-z][A-Za-z /-]+", text)
        ),
        "",
    )
    field_name = next(
        (
            text
            for text in texts
            if text != english_name and "欄位長度" not in text and not FIELD_CODE_PATTERN.search(text)
            and any("\u4e00" <= character <= "\u9fff" for character in text)
        ),
        "",
    )
    return {
        "field_code": code,
        "field_name": field_name,
        "field_english_name": english_name,
        "length": length_match.group(1) if length_match else "",
    }


def build_field_chunks(blocks: Iterable[LayoutBlock]) -> list[dict]:
    visible_blocks = [block for block in blocks if block.kind not in {"header", "footer"}]
    overview_rows = {}
    for block in visible_blocks:
        match = SUMMARY_ROW_PATTERN.fullmatch(block.text)
        if match:
            overview_rows[match.group(1)] = {
                "field_name": match.group(2),
                "field_english_name": match.group(3).strip(),
                "length": match.group(4),
                "data_type": match.group(5),
                "page": block.page,
            }

    code_indexes = [index for index, block in enumerate(visible_blocks) if FIELD_CODE_PATTERN.search(block.text)]
    field_chunks: list[dict] = []
    for position, code_index in enumerate(code_indexes):
        code_match = FIELD_CODE_PATTERN.search(visible_blocks[code_index].text)
        if not code_match:
            continue
        field_code = code_match.group(1)
        start = field_header_start(visible_blocks, code_index)
        end = field_header_start(visible_blocks, code_indexes[position + 1]) if position + 1 < len(code_indexes) else len(visible_blocks)
        header = visible_blocks[start : code_index + 1]
        section = visible_blocks[code_index + 1 : end]
        values = field_header_values(header, field_code)
        overview = overview_rows.get(field_code, {})
        for key in ("field_name", "field_english_name", "length"):
            if overview.get(key):
                values[key] = overview[key]
        if not values["field_name"]:
            continue

        token = field_code.replace(".", "_")
        summary_page = overview.get("page", header[0].page)
        summary_content = "\n".join(
            value
            for value in (
                f"欄位編號：{field_code}",
                f"欄位名稱：{values['field_name']}",
                f"英文欄位名稱：{values['field_english_name']}" if values["field_english_name"] else "",
                f"欄位長度：{values['length']}" if values["length"] else "",
                f"資料型態：{overview.get('data_type', '')}" if overview.get("data_type") else "",
            )
            if value
        )
        parent_chunk_id = f"field_{token}_detail_parent"
        field_chunks.append(
            {
                "chunk_id": f"field_{token}_summary",
                "page_number": summary_page,
                "page_start": summary_page,
                "page_end": summary_page,
                "block_type": "field_summary",
                "field_code": field_code,
                "field_name": values["field_name"],
                "field_english_name": values["field_english_name"],
                "detail_type": "",
                "parent_chunk_id": "",
                "content": summary_content,
            }
        )
        detail_positions = [index for index, block in enumerate(section) if detail_type_for(block.text)]
        if not detail_positions:
            continue
        detail_blocks = section[detail_positions[0] :]
        parent_content = " ".join(" ".join(block.text.split()) for block in detail_blocks)
        field_chunks.append(
            {
                "chunk_id": parent_chunk_id,
                "page_number": detail_blocks[0].page,
                "page_start": detail_blocks[0].page,
                "page_end": detail_blocks[-1].page,
                "block_type": "field_detail_parent",
                "field_code": field_code,
                "field_name": values["field_name"],
                "field_english_name": values["field_english_name"],
                "detail_type": "",
                "parent_chunk_id": "",
                "content": parent_content,
            }
        )
        for part_index, marker_index in enumerate(detail_positions):
            next_marker = detail_positions[part_index + 1] if part_index + 1 < len(detail_positions) else len(section)
            part_blocks = section[marker_index:next_marker]
            detail_type = detail_type_for(part_blocks[0].text)
            if not detail_type:
                continue
            field_chunks.append(
                {
                    "chunk_id": f"field_{token}_{detail_type}_{part_index + 1}",
                    "page_number": part_blocks[0].page,
                    "page_start": part_blocks[0].page,
                    "page_end": part_blocks[-1].page,
                    "block_type": "field_detail_part",
                    "field_code": field_code,
                    "field_name": values["field_name"],
                    "field_english_name": values["field_english_name"],
                    "detail_type": detail_type,
                    "parent_chunk_id": parent_chunk_id,
                    "content": " ".join(" ".join(block.text.split()) for block in part_blocks),
                }
            )
    return field_chunks


def parse_pdf(pdf_path: Path, *, max_pages: int | None, use_ocr: bool, force_ocr: bool, language: str, chunk_size: int, overlap: int) -> dict:
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

    field_chunks = build_field_chunks(layout_blocks)
    return {
        "source": str(pdf_path),
        "total_pages": len(document),
        "processed_pages": page_limit,
        "ocr_pages": sum(item["ocr_used"] for item in page_decisions),
        "ocr_required_pages": sum(item["ocr_required"] for item in page_decisions),
        "page_decisions": page_decisions,
        "layout_blocks": [asdict(block) for block in layout_blocks],
        "chunks": build_chunks(layout_blocks, chunk_size, overlap),
        "field_chunks": field_chunks,
    }


def write_pdf_chunk_report(report: dict, report_dir: Path, document_id: str) -> Path:
    """Write the parser output to a local JSON file for inspection."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{document_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def ingest_pdf(pdf_path: Path,*,document_id: str,filename: str,settings: Settings,report_dir: Path | None = None,) -> dict:
    try:
        report = parse_pdf(pdf_path,max_pages=None,use_ocr=True,force_ocr=False,language="chi_tra+eng",chunk_size=CHUNK_SIZE,overlap=CHUNK_OVERLAP,)
    except (OSError, RuntimeError) as error:
        raise RagServiceError("PDF parsing failed.", 400) from error

    report_path = None
    if report_dir:
        report_path = write_pdf_chunk_report(report, report_dir, document_id)

    chunks = [
        {
            **chunk,
            "page_start": chunk["page_number"],
            "page_end": chunk["page_number"],
            "block_type": "general_text",
            "field_code": "",
            "field_name": "",
            "field_english_name": "",
            "detail_type": "",
            "parent_chunk_id": "",
        }
        for chunk in report["chunks"]
        if chunk["content"].strip()
    ]
    chunks.extend(chunk for chunk in report["field_chunks"] if chunk["content"].strip())
    if not chunks:
        raise RagServiceError("No readable text was found in this PDF.", 400)

    client = weaviate_client(settings)
    try:
        collection = rag_collection(client)
        for index, chunk in enumerate(chunks, start=1):
            content = chunk["content"]
            collection.data.insert(
                properties={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": document_id,
                    "source_type": "pdf",
                    "url": "",
                    "title": filename,
                    "page_number": chunk["page_number"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "chunk_index": index,
                    "block_type": chunk["block_type"],
                    "field_code": chunk["field_code"],
                    "field_name": chunk["field_name"],
                    "field_english_name": chunk["field_english_name"],
                    "detail_type": chunk["detail_type"],
                    "parent_chunk_id": chunk["parent_chunk_id"],
                    "content": content,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                vector=embedding(content, settings),
            )
    except RagServiceError:
        raise
    except Exception as error:
        raise RagServiceError("Could not index the PDF in Weaviate.", 503) from error
    finally:
        client.close()

    return {
        "source_type": "pdf",
        "chunk_count": len(chunks),
        "field_chunk_count": len(report["field_chunks"]),
        "processed_pages": report["processed_pages"],
        "ocr_pages": report["ocr_pages"],
        "chunk_report_path": str(report_path) if report_path else None,
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
