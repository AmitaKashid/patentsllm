"""Command-line entry point for patent parsing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.io import list_pdf_files, write_jsonl
from patentllm.parsing.models import PageRecord, ParagraphRecord, ParseQualityReport, PatentChunk, PatentMetadata
from patentllm.parsing.parser import PatentPdfParser

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse patent PDFs into JSONL records.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw PDF patents.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSONL parser outputs.")
    parser.add_argument("--ocr-dpi", type=int, default=240, help="DPI used for OCR fallback.")
    parser.add_argument("--ocr-language", type=str, default="eng", help="Tesseract OCR language code.")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR fallback.")
    parser.add_argument("--disable-ocr-confidence", action="store_true", help="Skip confidence pass for faster parsing.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page limit for smoke tests.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--tessdata-dir",
        type=str,
        default=None,
        help="Path to Tesseract tessdata directory, for example C:\\Program Files\\Tesseract-OCR\\tessdata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = PatentParserConfig(
        ocr_enabled=not args.disable_ocr,
        ocr_language=args.ocr_language,
        ocr_dpi=args.ocr_dpi,
        max_pages=args.max_pages,
        tessdata_dir=args.tessdata_dir,
    )
    parser = PatentPdfParser(config=config)

    pdf_files = list_pdf_files(args.input_dir)
    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {args.input_dir}")

    all_metadata: list[PatentMetadata] = []
    all_pages: list[PageRecord] = []
    all_paragraphs: list[ParagraphRecord] = []
    all_chunks: list[PatentChunk] = []
    all_quality: list[ParseQualityReport] = []

    for pdf_file in pdf_files:
        LOGGER.info("Parsing %s", pdf_file.name)
        result = parser.parse_pdf(pdf_file)
        all_metadata.append(result.metadata)
        all_pages.extend(result.pages)
        all_paragraphs.extend(result.paragraphs)
        all_chunks.extend(result.chunks)
        all_quality.append(result.quality)
        LOGGER.info(
            "Parsed %s | pages=%d paragraphs=%d chunks=%d warnings=%d",
            pdf_file.name,
            len(result.pages),
            len(result.paragraphs),
            len(result.chunks),
            len(result.quality.warnings),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "documents": write_jsonl(args.output_dir / "documents.jsonl", all_metadata),
        "pages": write_jsonl(args.output_dir / "pages.jsonl", all_pages),
        "paragraphs": write_jsonl(args.output_dir / "paragraphs.jsonl", all_paragraphs),
        "chunks": write_jsonl(args.output_dir / "chunks.jsonl", all_chunks),
        "quality_report": write_jsonl(args.output_dir / "quality_report.jsonl", all_quality),
    }

    LOGGER.info("Wrote parser outputs to %s", args.output_dir)
    for name, count in counts.items():
        LOGGER.info("%s.jsonl: %d records", name, count)



if __name__ == "__main__":
    main()
