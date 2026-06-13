"""High-level patent PDF parser."""

from __future__ import annotations
from dataclasses import replace
import re
from pathlib import Path

from patentllm.parsing.chunker import PatentChunker
from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.metadata import extract_metadata
from patentllm.parsing.models import PageRecord, ParagraphRecord, ParseResult
from patentllm.parsing.pdf_text_extractor import PatentPdfTextExtractor
from patentllm.parsing.quality import build_quality_report
from patentllm.parsing.section_detection import detect_section_heading
from patentllm.parsing.text_cleaning import estimate_tokens, join_wrapped_lines, normalize_inline_spacing

_PARAGRAPH_MARKER_RE = re.compile(r"^\s*(?P<marker>\[?\d{4}\]?)\s*(?P<body>.*)$")
_FORMAL_CLAIM_HEADING_RE = re.compile(
    r"\b(?:CLAIMS?|WE CLAIM|I/WE CLAIM|WHAT IS CLAIMED)\b\s*:?",
    flags=re.IGNORECASE,
)

class PatentPdfParser:
    """Parse patent PDFs into structured metadata, pages, paragraphs, chunks, and quality records."""

    def __init__(self, config: PatentParserConfig | None = None) -> None:
        self.config = config or PatentParserConfig()
        self.extractor = PatentPdfTextExtractor(self.config)
        self.chunker = PatentChunker(self.config)

    def parse_pdf(self, pdf_path: Path) -> ParseResult:
        """Parse one patent PDF."""

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {pdf_path}")

        document_id = self._document_id(pdf_path)
        pages = self.extractor.extract_pages(pdf_path, document_id)
        metadata = extract_metadata(pdf_path, document_id, pages)
        paragraphs = self._build_paragraphs(pages)
        paragraphs = self._repair_missing_claim_section(pages, paragraphs)
        chunks = self.chunker.build_chunks(metadata, paragraphs)
        quality = build_quality_report(metadata, pages, paragraphs, chunks, self.config)

        return ParseResult(
            metadata=metadata,
            pages=pages,
            paragraphs=paragraphs,
            chunks=chunks,
            quality=quality,
        )

    def _build_paragraphs(self, pages: list[PageRecord]) -> list[ParagraphRecord]:
        paragraphs: list[ParagraphRecord] = []
        current_section = "front_page"
        sequence_index = 0

        for page in pages:
            page_paragraphs, current_section = self._paragraphs_from_page(
                page=page,
                initial_section=current_section,
                start_index=sequence_index,
            )
            paragraphs.extend(page_paragraphs)
            sequence_index += len(page_paragraphs)

        return paragraphs

    def _paragraphs_from_page(
        self,
        page: PageRecord,
        initial_section: str,
        start_index: int,
    ) -> tuple[list[ParagraphRecord], str]:
        section = initial_section
        paragraphs: list[ParagraphRecord] = []
        buffer: list[str] = []
        marker: str | None = None
        sequence_index = start_index

        def flush() -> None:
            nonlocal buffer, marker, sequence_index
            text = join_wrapped_lines(buffer)
            buffer = []
            if not text or len(text) < self.config.min_paragraph_chars:
                marker = None
                return

            paragraph_id = self._paragraph_id(page.document_id, page.page_number, sequence_index, marker)
            paragraphs.append(
                ParagraphRecord(
                    paragraph_id=paragraph_id,
                    document_id=page.document_id,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    sequence_index=sequence_index,
                    section=section,
                    text=text,
                    token_estimate=estimate_tokens(text),
                    marker=marker,
                )
            )
            sequence_index += 1
            marker = None

        for raw_line in page.text.splitlines():
            line = normalize_inline_spacing(raw_line)
            if not line:
                flush()
                continue

            detected_section = detect_section_heading(line)
            if detected_section:
                flush()
                section = detected_section
                continue

            marker_match = _PARAGRAPH_MARKER_RE.match(line)
            if marker_match:
                candidate_marker = marker_match.group("marker").strip("[]")
                body = marker_match.group("body").strip()
                if len(candidate_marker) == 4 and candidate_marker.isdigit():
                    flush()
                    marker = candidate_marker
                    if body:
                        buffer.append(body)
                    continue

            buffer.append(line)

        flush()
        return paragraphs, section

    @staticmethod
    def _document_id(pdf_path: Path) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", pdf_path.stem).strip("_")

    @staticmethod
    def _paragraph_id(document_id: str, page_number: int, sequence_index: int, marker: str | None) -> str:
        marker_part = marker if marker else f"seq{sequence_index:05d}"
        return f"{document_id}:p{page_number}:{marker_part}"
    

    def _repair_missing_claim_section(
        self,
        pages: list[PageRecord],
        paragraphs: list[ParagraphRecord],
    ) -> list[ParagraphRecord]:
        """Repair missed claim sections caused by OCR heading variants.

        Some patent PDFs use headings such as "We claim:" or "What is Claimed:"
        instead of a clean "CLAIMS" heading. OCR can also attach those headings
        to page headers. If no claims were detected, scan the final part of the
        document for a formal claim heading and reclassify later body paragraphs
        as claims.

        This is intentionally conservative:
        - It only runs when no claims section was detected.
        - It only searches the later part of the document.
        - It does not overwrite search-report or citation-list sections.
        """

        if any(paragraph.section == "claims" for paragraph in paragraphs):
            return paragraphs

        claim_start_page = self._find_formal_claim_start_page(pages)
        if claim_start_page is None:
            return paragraphs

        repaired: list[ParagraphRecord] = []
        protected_sections = {"search_report", "citation_list"}

        for paragraph in paragraphs:
            if (
                paragraph.page_number >= claim_start_page
                and paragraph.section not in protected_sections
            ):
                repaired.append(replace(paragraph, section="claims"))
            else:
                repaired.append(paragraph)

        return repaired

    @staticmethod
    def _find_formal_claim_start_page(pages: list[PageRecord]) -> int | None:
        """Find the first late-document page containing a formal claim heading."""

        if not pages:
            return None

        total_pages = max(page.page_number for page in pages)
        search_from_page = max(1, int(total_pages * 0.55))

        for page in pages:
            if page.page_number < search_from_page:
                continue

            if _FORMAL_CLAIM_HEADING_RE.search(page.text):
                return page.page_number

        return None
