"""Audit a generated PatentLLM report for structural and citation quality.

This is not model benchmarking.
It checks whether one generated report is usable enough for manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


REQUIRED_HEADINGS = [
    "# Patent Analysis for R&D",
    "## Scope snapshot",
    "## Main conclusion",
    "## Executive summary",
    "## Scope and evidence base",
    "## 1. Ownership and competitive landscape",
    "## 2. Temporal trends",
    "## 3. Technology deep dive and R&D relevance",
    "### 3.1 Polymer types",
    "### 3.2 Formulation strategies",
    "### 3.3 Application focus",
    "## 4. Key insights and R&D implications",
    "## 5. Limits of the conclusion",
    "## Patent key and extracted R&D signal",
]

TARGET_PATENTS = [
    "EP2756049A1",
    "EP2841522A1",
    "EP2411423A1",
    "EP2958970A1",
    "EP2686395A2",
    "EP3268442A1",
    "EP2456819A1",
    "EP3402857A1",
    "EP3707219A1",
    "EP4441161A1",
]

FORBIDDEN_LEGAL_TERMS = [
    "infringement",
    "freedom-to-operate",
    "fto",
    "validity",
    "patentability",
    "clearance",
    "opposition",
]


@dataclass(frozen=True, slots=True)
class AuditResult:
    check: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated PatentLLM report.")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/report_audits"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_text = args.report_path.read_text(encoding="utf-8")
    evidence_pack = json.loads(args.evidence_pack.read_text(encoding="utf-8"))

    valid_evidence_ids = {
        item["evidence_id"] for item in evidence_pack.get("evidence_items", [])
    }

    results = [
        check_required_headings(report_text),
        check_target_patents(report_text),
        check_citation_presence(report_text),
        check_citation_validity(report_text, valid_evidence_ids),
        check_forbidden_legal_terms(report_text),
        check_patent_key_rows(report_text),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    safe_report_name = args.report_path.stem
    csv_path = args.output_dir / f"{safe_report_name}_audit.csv"
    md_path = args.output_dir / f"{safe_report_name}_audit.md"

    write_csv(results, csv_path)
    write_markdown(results, md_path, args.report_path)

    passed = sum(result.passed for result in results)
    total = len(results)

    print(f"Audit complete: {passed}/{total} checks passed")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


def check_required_headings(report_text: str) -> AuditResult:
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in report_text]
    return AuditResult(
        check="required_headings",
        passed=not missing,
        detail="All required headings present." if not missing else f"Missing: {missing}",
    )


def check_target_patents(report_text: str) -> AuditResult:
    missing = [patent for patent in TARGET_PATENTS if patent not in report_text]
    return AuditResult(
        check="target_patents",
        passed=not missing,
        detail="All target patents mentioned." if not missing else f"Missing: {missing}",
    )


def check_citation_presence(report_text: str) -> AuditResult:
    citations = extract_citations(report_text)
    return AuditResult(
        check="citation_presence",
        passed=len(citations) >= 20,
        detail=f"Found {len(citations)} evidence citations.",
    )


def check_citation_validity(
    report_text: str,
    valid_evidence_ids: set[str],
) -> AuditResult:
    citations = set(extract_citations(report_text))
    invalid = sorted(citations - valid_evidence_ids)

    return AuditResult(
        check="citation_validity",
        passed=not invalid,
        detail="All citations are valid." if not invalid else f"Invalid citations: {invalid}",
    )


def check_forbidden_legal_terms(report_text: str) -> AuditResult:
    limits_start = report_text.lower().find("## 5. limits of the conclusion")
    before_limits = report_text[:limits_start].lower() if limits_start != -1 else report_text.lower()

    found = [
        term for term in FORBIDDEN_LEGAL_TERMS
        if term in before_limits
    ]

    return AuditResult(
        check="legal_term_leakage_before_limits",
        passed=not found,
        detail="No legal terms before limitations section." if not found else f"Found before limits: {found}",
    )


def check_patent_key_rows(report_text: str) -> AuditResult:
    patent_key_start = report_text.find("## Patent key and extracted R&D signal")
    patent_key = report_text[patent_key_start:] if patent_key_start != -1 else ""

    count = sum(1 for patent in TARGET_PATENTS if patent in patent_key)

    return AuditResult(
        check="patent_key_completeness",
        passed=count == len(TARGET_PATENTS),
        detail=f"Patent key contains {count}/{len(TARGET_PATENTS)} target patents.",
    )


def extract_citations(report_text: str) -> list[str]:
    return re.findall(r"\[(E\d+)\]", report_text)


def write_csv(results: list[AuditResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_markdown(
    results: list[AuditResult],
    path: Path,
    report_path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# Generated Report Audit\n\n")
        file.write(f"Report: `{report_path}`\n\n")
        file.write("| Check | Passed | Detail |\n")
        file.write("|---|---:|---|\n")

        for result in results:
            detail = result.detail.replace("|", "\\|")
            file.write(f"| {result.check} | {result.passed} | {detail} |\n")


if __name__ == "__main__":
    main()