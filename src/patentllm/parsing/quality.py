"""Quality reporting for parsed patent PDFs."""

from __future__ import annotations

from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import PageRecord, ParagraphRecord, ParseQualityReport, PatentChunk, PatentMetadata
from patentllm.parsing.section_detection import is_evidence_section


def build_quality_report(
    metadata: PatentMetadata,
    pages: list[PageRecord],
    paragraphs: list[ParagraphRecord],
    chunks: list[PatentChunk],
    config: PatentParserConfig,
) -> ParseQualityReport:
    """Create a compact document-level parse quality report."""

    detected_sections = sorted({paragraph.section for paragraph in paragraphs})
    ocr_pages = [page for page in pages if page.extraction_method == "tesseract_ocr"]
    native_pages = [page for page in pages if page.extraction_method == "native_pdf_text"]
    low_confidence_pages = [
        page.page_number
        for page in ocr_pages
        if page.ocr_confidence_mean is not None
        and page.ocr_confidence_mean < config.low_ocr_confidence_threshold
    ]
    warnings = _build_warnings(
        metadata=metadata,
        pages=pages,
        paragraphs=paragraphs,
        chunks=chunks,
        low_confidence_pages=low_confidence_pages,
    )

    return ParseQualityReport(
        document_id=metadata.document_id,
        source_file=metadata.source_file,
        page_count=metadata.page_count,
        pages_extracted=len(pages),
        pages_with_text=sum(1 for page in pages if page.char_count > 0),
        ocr_pages=len(ocr_pages),
        native_text_pages=len(native_pages),
        low_confidence_ocr_pages=low_confidence_pages,
        paragraph_count=len(paragraphs),
        chunk_count=len(chunks),
        detected_sections=detected_sections,
        title_found=bool(metadata.title),
        abstract_found=bool(metadata.abstract),
        claims_found="claims" in detected_sections,
        examples_found=any(is_evidence_section(section) for section in detected_sections),
        tables_detected=bool(metadata.detected_tables),
        figures_detected=bool(metadata.detected_figures),
        warnings=warnings,
    )


def _build_warnings(
    metadata: PatentMetadata,
    pages: list[PageRecord],
    paragraphs: list[ParagraphRecord],
    chunks: list[PatentChunk],
    low_confidence_pages: list[int],
) -> list[str]:
    warnings: list[str] = []

    if not pages:
        warnings.append("No pages were extracted.")
    if pages and not any(page.char_count > 0 for page in pages):
        warnings.append("No extractable text found on any page.")
    if not metadata.title:
        warnings.append("Title was not detected.")
    if not metadata.abstract:
        warnings.append("Abstract was not detected; this is common when cover-page OCR is noisy.")
    if not metadata.ipc_cpc_classifications:
        warnings.append("IPC/CPC classifications were not detected as structured fields.")
    if not paragraphs:
        warnings.append("No paragraphs were produced.")
    if chunks and not any(chunk.chunk_type == "title_abstract_claims" for chunk in chunks):
        warnings.append("Title/abstract/claims document-view chunk was not produced.")
    if not any(paragraph.section == "claims" for paragraph in paragraphs):
        warnings.append("Claims section was not detected in parsed pages; it may appear after max_pages or require better OCR.")
    if low_confidence_pages:
        warnings.append(f"Low OCR confidence on pages: {low_confidence_pages}")
    if metadata.detected_tables:
        warnings.append("Tables were detected as text references, but table structure is not yet extracted.")
    if metadata.detected_figures:
        warnings.append("Figures were detected as text references, but figure images are not yet extracted.")

    return warnings
