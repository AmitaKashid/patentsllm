from pathlib import Path


def test_final_pipeline_documentation_exists() -> None:
    expected_files = [
        Path("docs/architecture/pipeline_design.md"),
        Path("docs/results/final_claim_bank_grounded_report.md"),
        Path("docs/results/claim_bank_alignment_summary.md"),
    ]

    for file_path in expected_files:
        assert file_path.exists(), f"Missing documentation artifact: {file_path}"


def test_claim_bank_alignment_summary_records_final_result() -> None:
    summary_path = Path("docs/results/claim_bank_alignment_summary.md")
    text = summary_path.read_text(encoding="utf-8")

    assert "Rows checked: 55" in text
    assert "Aligned or partially aligned: 55/55" in text