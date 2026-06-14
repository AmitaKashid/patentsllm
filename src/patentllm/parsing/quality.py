"""Quality reporting for parsed patent PDFs."""

from __future__ import annotations

from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import PageRecord, ParagraphRecord, ParseQualityReport, PatentChunk, PatentMetadata
from patentllm.parsing.section_detection import is_evidence_section


_CLAIM_CHUNK_TYPES = {"claim", "claim_clause_window"}


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

    claim_chunks = [chunk for chunk in chunks if chunk.chunk_type in _CLAIM_CHUNK_TYPES]
    claim_numbers = sorted(
        {
            int(chunk.metadata["claim_number"])
            for chunk in claim_chunks
            if chunk.metadata.get("claim_number") is not None
        }
    )
    missing_claim_numbers = _missing_numbers(claim_numbers)
    low_confidence_claims = [
        chunk
        for chunk in claim_chunks
        if chunk.metadata.get("claim_confidence") == "low"
        or chunk.metadata.get("claim_warnings")
    ]

    warnings = _build_warnings(
        metadata=metadata,
        pages=pages,
        paragraphs=paragraphs,
        chunks=chunks,
        low_confidence_pages=low_confidence_pages,
        claim_numbers=claim_numbers,
        missing_claim_numbers=missing_claim_numbers,
        low_confidence_claim_count=len(low_confidence_claims),
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
        embeddable_chunk_count=sum(1 for chunk in chunks if chunk.embeddable),
        detected_sections=detected_sections,
        title_found=bool(metadata.title),
        abstract_found=bool(metadata.abstract),
        claims_found=bool(claim_chunks),
        claim_count=len(claim_numbers),
        claim_numbers_detected=claim_numbers,
        missing_claim_numbers=missing_claim_numbers,
        low_confidence_claim_count=len(low_confidence_claims),
        examples_found=any(is_evidence_section(section) for section in detected_sections),
        tables_detected=bool(metadata.detected_tables),
        figures_detected=bool(metadata.detected_figures),
        suspicious_chunk_count=sum(1 for chunk in chunks if "suspicious_ocr" in chunk.quality_flags),
        warnings=warnings,
    )


def _build_warnings(
    metadata: PatentMetadata,
    pages: list[PageRecord],
    paragraphs: list[ParagraphRecord],
    chunks: list[PatentChunk],
    low_confidence_pages: list[int],
    claim_numbers: list[int],
    missing_claim_numbers: list[int],
    low_confidence_claim_count: int,
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
    if metadata.metadata_warnings:
        warnings.extend(metadata.metadata_warnings)
    if not paragraphs:
        warnings.append("No paragraphs were produced.")
    if chunks and not any(chunk.chunk_type == "title_abstract_claims" for chunk in chunks):
        warnings.append("Title/abstract/claims document-view chunk was not produced.")
    if not any(chunk.chunk_type in _CLAIM_CHUNK_TYPES for chunk in chunks):
        warnings.append("No claim chunks were produced from the claims section.")
    if claim_numbers and claim_numbers[0] != 1:
        warnings.append(f"Claim sequence starts at {claim_numbers[0]}, expected claim 1.")
    if missing_claim_numbers:
        warnings.append(f"Claim sequence has missing claim numbers: {missing_claim_numbers}.")
    if low_confidence_claim_count:
        warnings.append(f"Low/medium-confidence claim extraction issues detected in {low_confidence_claim_count} claim chunks.")
    if low_confidence_pages:
        warnings.append(f"Low OCR confidence on pages: {low_confidence_pages}")
    if metadata.detected_tables:
        warnings.append("Tables were detected as text references; table-heavy OCR chunks are flagged and may be excluded from embedding.")
    if metadata.detected_figures:
        warnings.append("Figures were detected as text references, but figure images are not yet extracted.")
    if any("excluded_from_embedding_due_to_table_ocr_noise" in chunk.quality_flags for chunk in chunks):
        warnings.append("Some table-heavy OCR chunks were marked non-embeddable to protect retrieval quality.")

    return warnings


def _missing_numbers(numbers: list[int]) -> list[int]:
    if not numbers:
        return []
    return [value for value in range(1, numbers[-1] + 1) if value not in set(numbers)]
