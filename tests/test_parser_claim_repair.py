from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import PageRecord, ParagraphRecord
from patentllm.parsing.parser import PatentPdfParser


def test_repair_missing_claim_section_from_we_claim_heading() -> None:
    parser = PatentPdfParser(PatentParserConfig())

    pages = [
        PageRecord(
            document_id="DOC",
            source_file="DOC.pdf",
            page_number=1,
            text="Description text",
            extraction_method="tesseract_ocr",
            char_count=16,
        ),
        PageRecord(
            document_id="DOC",
            source_file="DOC.pdf",
            page_number=10,
            text="WO 0000/000000 PCT/XX0000/000000 We claim: 1. A composition comprising...",
            extraction_method="tesseract_ocr",
            char_count=80,
        ),
    ]

    paragraphs = [
        ParagraphRecord(
            paragraph_id="p1",
            document_id="DOC",
            source_file="DOC.pdf",
            page_number=1,
            sequence_index=0,
            section="detailed_description",
            text="Description text",
            token_estimate=5,
            marker=None,
        ),
        ParagraphRecord(
            paragraph_id="p10",
            document_id="DOC",
            source_file="DOC.pdf",
            page_number=10,
            sequence_index=1,
            section="detailed_description",
            text="1. A composition comprising a polymer.",
            token_estimate=8,
            marker=None,
        ),
    ]

    repaired = parser._repair_missing_claim_section(pages, paragraphs)

    assert repaired[0].section == "detailed_description"
    assert repaired[1].section == "claims"


def test_repair_does_not_overwrite_existing_claims() -> None:
    parser = PatentPdfParser(PatentParserConfig())

    pages = [
        PageRecord(
            document_id="DOC",
            source_file="DOC.pdf",
            page_number=10,
            text="We claim: 1. A composition comprising...",
            extraction_method="tesseract_ocr",
            char_count=40,
        )
    ]

    paragraphs = [
        ParagraphRecord(
            paragraph_id="p10",
            document_id="DOC",
            source_file="DOC.pdf",
            page_number=10,
            sequence_index=0,
            section="claims",
            text="1. A composition comprising a polymer.",
            token_estimate=8,
            marker=None,
        )
    ]

    repaired = parser._repair_missing_claim_section(pages, paragraphs)

    assert repaired == paragraphs