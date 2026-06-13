from patentllm.parsing.section_detection import detect_section_heading


def test_detects_standard_claim_heading() -> None:
    assert detect_section_heading("CLAIMS") == "claims"


def test_detects_us_style_claim_headings() -> None:
    assert detect_section_heading("We claim:") == "claims"
    assert detect_section_heading("I/We claim:") == "claims"
    assert detect_section_heading("What is Claimed:") == "claims"


def test_rejects_numbered_claim_as_section_heading() -> None:
    assert detect_section_heading("1. A hot melt adhesive composition comprising:") is None


def test_rejects_ocr_noise_as_section_heading() -> None:
    assert detect_section_heading("j el i qls 2 5i bet") is None
    assert detect_section_heading("S SZALE BI 2 I ZI I GL5 G OSGG GL53 3 S") is None