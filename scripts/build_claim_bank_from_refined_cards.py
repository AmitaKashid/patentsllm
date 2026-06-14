"""Build an atomic claim bank from repaired refined patent cards.

Why this exists:
The report generator was attaching valid evidence IDs to unsupported or weakly
related claims. This script creates deterministic, field-level claim units so
the report generator can cite only pre-approved claim/evidence pairs.

Input:
    patent_cards_refined_repaired.json

Output:
    data/processed/claim_banks/<run_id>/
    ├── claim_bank.json
    ├── claim_bank.md
    └── claim_bank_quality.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


FIELD_TO_REPORT_SECTION = {
    "core_technology": "technology_deep_dive",
    "polymer_route": "technology_deep_dive",
    "formulation_strategy": "formulation_strategies",
    "application_focus": "application_focus",
    "performance_evidence": "rd_implications",
    "rd_relevance": "rd_implications",
}


FIELD_TO_CLAIM_TYPE = {
    "core_technology": "core_technology",
    "polymer_route": "polymer_route",
    "formulation_strategy": "formulation_strategy",
    "application_focus": "application_focus",
    "performance_evidence": "performance_evidence",
    "rd_relevance": "rd_relevance",
}


@dataclass(frozen=True, slots=True)
class ClaimBankItem:
    claim_id: str
    patent_publication: str
    owner: str
    priority_year: str
    claim_type: str
    report_section: str
    claim_text: str
    evidence_ids: list[str]
    approved_for_report: bool
    rejection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build atomic claim bank from repaired refined patent cards."
    )

    parser.add_argument("--cards-json", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/claim_banks"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cards = load_cards(args.cards_json)
    claims = build_claim_bank(cards)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "claim_bank.json"
    md_path = output_dir / "claim_bank.md"
    quality_path = output_dir / "claim_bank_quality.csv"

    json_path.write_text(
        json.dumps([asdict(claim) for claim in claims], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown(claims, md_path)
    write_quality_csv(claims, quality_path)

    print(f"Wrote claim bank to: {output_dir}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Quality: {quality_path}")


def load_cards(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cards JSON not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise TypeError("Expected cards JSON to contain a list.")

    return data


def build_claim_bank(cards: list[dict[str, Any]]) -> list[ClaimBankItem]:
    claims: list[ClaimBankItem] = []
    claim_index = 1

    for card in cards:
        field_map = card.get("field_evidence_map", {})
        if not isinstance(field_map, dict):
            field_map = {}

        for field, claim_type in FIELD_TO_CLAIM_TYPE.items():
            claim_text = clean_text(str(card.get(field, "")))
            evidence_ids = normalize_evidence_ids(field_map.get(field, []))

            approved, rejection_reason = validate_claim(
                claim_text=claim_text,
                evidence_ids=evidence_ids,
            )

            claims.append(
                ClaimBankItem(
                    claim_id=f"C{claim_index:03d}",
                    patent_publication=str(card.get("patent_publication", "")).strip(),
                    owner=str(card.get("owner", "")).strip(),
                    priority_year=str(card.get("priority_year", "")).strip(),
                    claim_type=claim_type,
                    report_section=FIELD_TO_REPORT_SECTION[field],
                    claim_text=claim_text,
                    evidence_ids=evidence_ids,
                    approved_for_report=approved,
                    rejection_reason=rejection_reason,
                )
            )

            claim_index += 1

    return claims


def validate_claim(
    claim_text: str,
    evidence_ids: list[str],
) -> tuple[bool, str]:
    if not claim_text:
        return False, "empty claim text"

    if claim_text.lower() == "not available in supplied evidence":
        return False, "field unavailable in supplied evidence"

    if not evidence_ids:
        return False, "missing evidence ids"

    if len(evidence_ids) > 3:
        return False, "too many evidence ids for atomic claim"

    if looks_too_generic(claim_text):
        return False, "claim text too generic"

    return True, ""


def looks_too_generic(text: str) -> bool:
    lowered = text.lower()

    generic_phrases = [
        "optimize adhesive performance",
        "improve adhesive performance",
        "application versatility",
        "industrial applications",
        "potential applications",
        "further research is needed",
    ]

    return any(phrase in lowered for phrase in generic_phrases)


def normalize_evidence_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return []


def clean_text(text: str) -> str:
    return " ".join(text.split())


def write_markdown(claims: list[ClaimBankItem], path: Path) -> None:
    lines = ["# Claim Bank", ""]

    approved_claims = [claim for claim in claims if claim.approved_for_report]
    rejected_claims = [claim for claim in claims if not claim.approved_for_report]

    lines.append(f"- Total claims: {len(claims)}")
    lines.append(f"- Approved claims: {len(approved_claims)}")
    lines.append(f"- Rejected claims: {len(rejected_claims)}")
    lines.append("")

    for claim in claims:
        status = "APPROVED" if claim.approved_for_report else "REJECTED"
        evidence = ", ".join(f"[{eid}]" for eid in claim.evidence_ids)

        lines.append(f"## {claim.claim_id} — {status}")
        lines.append("")
        lines.append(f"- Patent: {claim.patent_publication}")
        lines.append(f"- Owner: {claim.owner}")
        lines.append(f"- Priority year: {claim.priority_year}")
        lines.append(f"- Claim type: {claim.claim_type}")
        lines.append(f"- Report section: {claim.report_section}")
        lines.append(f"- Claim: {claim.claim_text}")
        lines.append(f"- Evidence: {evidence}")
        if claim.rejection_reason:
            lines.append(f"- Rejection reason: {claim.rejection_reason}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_csv(claims: list[ClaimBankItem], path: Path) -> None:
    fieldnames = [
        "claim_id",
        "patent_publication",
        "owner",
        "priority_year",
        "claim_type",
        "report_section",
        "approved_for_report",
        "rejection_reason",
        "evidence_count",
        "claim_text",
        "evidence_ids",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for claim in claims:
            writer.writerow(
                {
                    "claim_id": claim.claim_id,
                    "patent_publication": claim.patent_publication,
                    "owner": claim.owner,
                    "priority_year": claim.priority_year,
                    "claim_type": claim.claim_type,
                    "report_section": claim.report_section,
                    "approved_for_report": claim.approved_for_report,
                    "rejection_reason": claim.rejection_reason,
                    "evidence_count": len(claim.evidence_ids),
                    "claim_text": claim.claim_text,
                    "evidence_ids": ", ".join(claim.evidence_ids),
                }
            )


if __name__ == "__main__":
    main()