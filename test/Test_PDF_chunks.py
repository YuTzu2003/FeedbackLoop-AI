"""Index a PDF with the Flask application's Elasticsearch retrieval pipeline."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from pipeline.load_pdf import ingest_pdf
from services.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a PDF in Elasticsearch.")
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF does not exist: {args.pdf}")

    load_dotenv()
    result = ingest_pdf(
        args.pdf,
        document_id=args.pdf.stem,
        filename=args.pdf.name,
        settings=load_settings(),
    )
    print(f"Indexed {result['chunk_count']} chunks from {args.pdf.name}.")


if __name__ == "__main__":
    main()
