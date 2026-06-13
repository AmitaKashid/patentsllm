from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import PageRecord, ParagraphRecord, PatentChunk, PatentMetadata
from patentllm.parsing.quality import build_quality_report


def test_quality_report_flags_missing_claims_and_low_confidence() -> None:
    metadata = PatentMetadata(
        document_id="WO_TEST_A1",
        source_file="WO_TEST_A1.pdf",
        page_count=1,
        title="TEST TITLE",
    )
    pages = [
        PageRecord(
            document_id="WO_TEST_A1",
            source_file="WO_TEST_A1.pdf",
            page_number=1,
            text="Some OCR text",
            extraction_method="tesseract_ocr",
            char_count=13,
            ocr_confidence_mean=40.0,
            ocr_word_count=3,
        )
    ]
    paragraphs = [
        ParagraphRecord(
            paragraph_id="p1",
            document_id="WO_TEST_A1",
            source_file="WO_TEST_A1.pdf",
            page_number=1,
            sequence_index=1,
            section="technical_field",
            text="Some paragraph",
            token_estimate=3,
        )
    ]
    chunks = [
        PatentChunk(
            chunk_id="c1",
            document_id="WO_TEST_A1",
            source_file="WO_TEST_A1.pdf",
            publication_number=None,
            title="TEST TITLE",
            section="metadata",
            chunk_type="metadata",
            retrieval_tier="metadata",
            page_start=1,
            page_end=1,
            text="metadata",
            token_estimate=1,
            embeddable=False,
        )
    ]
    report = build_quality_report(metadata, pages, paragraphs, chunks, PatentParserConfig())
    assert report.low_confidence_ocr_pages == [1]
    assert not report.claims_found
    assert report.embeddable_chunk_count == 0
    assert report.warnings


def test_quality_report_uses_claim_metadata_numbers() -> None:
    metadata = PatentMetadata(
        document_id="WO_TEST_A1",
        source_file="WO_TEST_A1.pdf",
        page_count=1,
        title="TEST TITLE",
        abstract="An adhesive.",
    )
    pages = []
    paragraphs = []
    chunks = [
        PatentChunk(
            chunk_id="c1",
            document_id="WO_TEST_A1",
            source_file="WO_TEST_A1.pdf",
            publication_number=None,
            title="TEST TITLE",
            section="claims",
            chunk_type="claim_clause_window",
            retrieval_tier="child",
            page_start=1,
            page_end=1,
            text="1. A composition comprising a polymer.",
            token_estimate=10,
            metadata={"claim_number": 1, "claim_confidence": "high"},
        )
    ]
    report = build_quality_report(metadata, pages, paragraphs, chunks, PatentParserConfig())
    assert report.claims_found
    assert report.claim_numbers_detected == [1]
