from patentllm.parsing.claim_extraction import extract_claims_from_paragraphs
from patentllm.parsing.models import ParagraphRecord
from patentllm.parsing.text_cleaning import estimate_tokens


def _paragraph(text: str, page: int = 10, sequence_index: int = 1) -> ParagraphRecord:
    return ParagraphRecord(
        paragraph_id=f"p{page}_{sequence_index}",
        document_id="WO_TEST_A1",
        source_file="WO_TEST_A1.pdf",
        page_number=page,
        sequence_index=sequence_index,
        section="claims",
        text=text,
        token_estimate=estimate_tokens(text),
    )


def test_claim_extractor_discards_pre_claim_noise() -> None:
    result = extract_claims_from_paragraphs(
        [
            _paragraph(
                "20. Random search report noise.\n"
                "18. More table noise.\n"
                "1. A hot melt adhesive composition comprising a polymer and a tackifier.\n"
                "2. The composition according to claim 1, wherein the polymer is a propylene polymer."
            )
        ]
    )

    assert [claim.claim_number for claim in result.claims] == [1, 2]
    assert result.claims[0].text.startswith("1.")


def test_claim_extractor_merges_non_increasing_candidates() -> None:
    result = extract_claims_from_paragraphs(
        [
            _paragraph(
                "1. An adhesive composition comprising a polymer.\n"
                "1. Duplicate OCR line that should not become a second claim.\n"
                "2. The adhesive composition of claim 1, wherein the polymer is olefinic."
            )
        ]
    )

    assert [claim.claim_number for claim in result.claims] == [1, 2]
    assert "Duplicate OCR" in result.claims[0].text


def test_claim_extractor_ignores_early_search_report_mentions_before_real_claims() -> None:
    result = extract_claims_from_paragraphs(
        [
            _paragraph("Published with international search report (Art. 21(3)).", page=1),
            _paragraph("1. A hot melt adhesive composition comprising a polymer and a tackifier.", page=42),
            _paragraph("2. The composition of claim 1 wherein the polymer is propylene-based.", page=42),
            _paragraph("INTERNATIONAL SEARCH REPORT International application No.", page=46),
        ]
    )

    assert [claim.claim_number for claim in result.claims] == [1, 2]


def test_claim_extractor_infers_missing_claim_one_from_preceding_fragments() -> None:
    result = extract_claims_from_paragraphs(
        [
            _paragraph("(A) a propylene homopolymer having a melting point of 100°C", page=51),
            _paragraph("5 or lower which is obtainable by polymerizing propylene using a catalyst", page=51),
            _paragraph("2. The hot melt adhesive according to claim 1, wherein the copolymer is ethylene-based.", page=51),
        ]
    )

    assert [claim.claim_number for claim in result.claims] == [1, 2]
    assert result.claims[0].confidence == "low"
