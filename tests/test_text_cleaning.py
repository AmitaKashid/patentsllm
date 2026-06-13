from patentllm.parsing.text_cleaning import estimate_tokens, join_wrapped_lines, normalize_page_text


def test_normalize_page_text_removes_short_patent_headers() -> None:
    text = "WO 2023/099461\nPCT/EP2022/083649\nActual invention text remains."
    assert normalize_page_text(text) == "Actual invention text remains."


def test_join_wrapped_lines_repairs_soft_hyphenation() -> None:
    assert join_wrapped_lines(["poly-", "propylene composition"]) == "polypropylene composition"


def test_estimate_tokens_returns_positive_count() -> None:
    assert estimate_tokens("hot melt adhesive composition") >= 4

from patentllm.parsing.section_detection import detect_section_heading


def test_detects_known_patent_sections() -> None:
    assert detect_section_heading("BACKGROUND OF THE INVENTION") == "background"
    assert detect_section_heading("DETAILED DESCRIPTION OF THE INVENTION") == "detailed_description"
    assert detect_section_heading("CLAIMS") == "claims"


def test_rejects_ocr_noise_as_section_heading() -> None:
    assert detect_section_heading("j el i qls 2 5i bet") is None
    assert detect_section_heading("S SZALE BI 2 I ZI I GL5 G OSGG GL53 3 S") is None
    assert detect_section_heading("A preferred tackifier is a hydrogenated aromatic modified dicyclopentadiene resin with a") is None


def test_does_not_treat_numbered_claim_as_section() -> None:
    assert detect_section_heading("1. A hot melt adhesive composition comprising:") is None