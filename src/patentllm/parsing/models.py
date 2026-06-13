"""Data models produced by the parsing layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class PatentMetadata:
    """Document-level metadata extracted from filename, cover page, and body text."""

    document_id: str
    source_file: str
    page_count: int
    publication_number: str | None = None
    kind_code: str | None = None
    title: str | None = None
    abstract: str | None = None
    applicant: str | None = None
    inventors: str | None = None
    publication_date: str | None = None
    application_number: str | None = None
    filing_date: str | None = None
    priority_date: str | None = None
    ipc_cpc_classifications: list[str] = field(default_factory=list)
    cited_patent_literature: list[str] = field(default_factory=list)
    detected_tables: list[str] = field(default_factory=list)
    detected_figures: list[str] = field(default_factory=list)
    raw_cover_fields: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PageRecord:
    """Page-level extraction result."""

    document_id: str
    source_file: str
    page_number: int
    text: str
    extraction_method: str
    char_count: int
    ocr_confidence_mean: float | None = None
    ocr_word_count: int | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParagraphRecord:
    """Patent paragraph or paragraph-like block."""

    paragraph_id: str
    document_id: str
    source_file: str
    page_number: int
    sequence_index: int
    section: str
    text: str
    token_estimate: int
    marker: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PatentChunk:
    """Embedding-ready or synthesis-ready chunk."""

    chunk_id: str
    document_id: str
    source_file: str
    publication_number: str | None
    title: str | None
    section: str
    chunk_type: str
    retrieval_tier: str
    page_start: int | None
    page_end: int | None
    text: str
    token_estimate: int
    paragraph_ids: list[str] = field(default_factory=list)
    source_page_numbers: list[int] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParseQualityReport:
    """Document-level parsing quality report."""

    document_id: str
    source_file: str
    page_count: int
    pages_extracted: int
    pages_with_text: int
    ocr_pages: int
    native_text_pages: int
    low_confidence_ocr_pages: list[int]
    paragraph_count: int
    chunk_count: int
    detected_sections: list[str]
    title_found: bool
    abstract_found: bool
    claims_found: bool
    examples_found: bool
    tables_detected: bool
    figures_detected: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Complete parser output for one patent PDF."""

    metadata: PatentMetadata
    pages: list[PageRecord]
    paragraphs: list[ParagraphRecord]
    chunks: list[PatentChunk]
    quality: ParseQualityReport
