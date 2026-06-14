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
            keep_short_claim_fragment = (
                section == "claims"
                and bool(text)
                and (
                    bool(re.match(r"^\s*\d{1,3}\s*[\.)]", text))
                    or bool(re.search(r"\b(?:comprising|wherein|according to claim|Claim \d+|A hot melt adhesive|A composition|A method)\b", text, flags=re.IGNORECASE))
                    or text.startswith(("(A)", "(B)", "(C)", "(D)", "(f)", "(g)", "(h)", "(i)", "(j)"))
                )
            )
            if not text or (len(text) < self.config.min_paragraph_chars and not keep_short_claim_fragment):
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
        """Recover claim sections when OCR misses the formal CLAIMS heading.

        Some scanned PCT PDFs do not expose a clean "CLAIMS" heading after OCR.
        In those cases, the parser may classify late claim paragraphs as
        detailed_description. This method repairs only that situation.

        Rules:
        - If claim paragraphs already exist, return unchanged.
        - Look for late pages containing "We claim" / "Claims" / numbered claim starts.
        - Mark matching paragraphs on those pages as claims.
        - Stop before the International Search Report.
        """

        if any(paragraph.section == "claims" for paragraph in paragraphs):
            return paragraphs

        claim_page_numbers = self._detect_claim_page_numbers(pages)
        if not claim_page_numbers:
            return paragraphs

        repaired: list[ParagraphRecord] = []
        claim_mode = False

        for paragraph in paragraphs:
            if paragraph.page_number not in claim_page_numbers:
                repaired.append(paragraph)
                continue

            text = paragraph.text.strip()

            if self._is_search_report_text(text):
                claim_mode = False
                repaired.append(paragraph)
                continue

            if self._looks_like_claim_start(text):
                claim_mode = True
                repaired.append(replace(paragraph, section="claims"))
                continue

            if claim_mode and self._looks_like_claim_continuation(text):
                repaired.append(replace(paragraph, section="claims"))
                continue

            repaired.append(paragraph)

        return repaired

    @staticmethod
    def _detect_claim_page_numbers(pages: list[PageRecord]) -> set[int]:
        """Detect likely claim pages from page-level OCR text."""

        claim_pages: set[int] = set()

        for page in pages:
            text = page.text or ""
            lowered = text.lower()

            if "international search report" in lowered:
                continue

            has_claim_heading = (
                "we claim" in lowered
                or "\nclaims\n" in lowered
                or lowered.strip().startswith("claims")
            )
            has_numbered_claim = bool(
                re.search(
                    r"(?:^|\n|\s)(?:1|2|3)\s*[\.)]\s+"
                    r"(?:a|an|the|use|method|composition|adhesive|laminate|article)\b",
                    lowered,
                    flags=re.IGNORECASE,
                )
            )

            if has_claim_heading or has_numbered_claim:
                claim_pages.add(page.page_number)

        return claim_pages

    @staticmethod
    def _looks_like_claim_start(text: str) -> bool:
        """Return True if a paragraph looks like the start of a patent claim."""

        normalized = " ".join(text.split())
        return bool(
            re.match(
                r"^\s*\d{1,3}\s*[\.)]\s+"
                r"(?:A|An|The|Use|Method|Composition|Adhesive|Laminate|Article)\b",
                normalized,
            )
        )

    @staticmethod
    def _looks_like_claim_continuation(text: str) -> bool:
        """Return True if a paragraph likely continues a previous claim."""

        normalized = " ".join(text.split())
        if not normalized:
            return False

        if re.match(r"^\s*\d{1,3}\s*[\.)]\s+", normalized):
            return False

        continuation_markers = (
            "wherein",
            "comprising",
            "consisting",
            "selected from",
            "about",
            "from",
            "and",
            "or",
            "(a)",
            "(b)",
            "(c)",
            "(d)",
            "(e)",
            "(f)",
            "(g)",
            "(h)",
            "(i)",
            "(j)",
        )

        return normalized.lower().startswith(continuation_markers)

    @staticmethod
    def _is_search_report_text(text: str) -> bool:
        """Return True if text belongs to the International Search Report."""

        lowered = text.lower()
        return (
            "international search report" in lowered
            or "documents considered to be relevant" in lowered
            or "form pct/isa/210" in lowered
        )
