"""Repair high-risk report tables using approved claim-bank items.

Why:
The claim-bank report passed structural audit, but citation-support audit showed
that most remaining problems are markdown table rows with broad multi-citation
claims. This script replaces those high-risk sections with deterministic,
claim-bank-grounded tables.

Input:
    final_report_*.md
    claim_bank.json

Output:
    *_deterministic_tables.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_PUBLICATIONS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace high-risk report tables with deterministic claim-bank tables."
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--claim-bank-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_text = args.report_path.read_text(encoding="utf-8")
    claims = load_approved_claims(args.claim_bank_json)

    repaired = report_text

    repaired = replace_section(
        text=repaired,
        start_heading="## 1. Ownership and competitive landscape",
        end_heading="## 2. Temporal trends",
        replacement=build_ownership_section(claims),
    )

    repaired = replace_section(
        text=repaired,
        start_heading="## 2. Temporal trends",
        end_heading="## 3. Technology deep dive and R&D relevance",
        replacement=build_temporal_section(claims),
    )

    repaired = replace_subsection(
        text=repaired,
        start_heading="### 3.2 Formulation strategies",
        end_heading="### 3.3 Application focus",
        replacement=build_formulation_subsection(claims),
    )

    repaired = replace_subsection(
        text=repaired,
        start_heading="### 3.3 Application focus",
        end_heading="## 4. Key insights and R&D implications",
        replacement=build_application_subsection(claims),
    )

    repaired = replace_section(
        text=repaired,
        start_heading="## 4. Key insights and R&D implications",
        end_heading="## 5. Limits of the conclusion",
        replacement=build_rd_implications_section(claims),
    )

    output_path = args.output_path
    if output_path is None:
        output_path = args.report_path.with_name(
            args.report_path.stem + "_deterministic_tables.md"
        )

    output_path.write_text(repaired.strip() + "\n", encoding="utf-8")
    print(f"Wrote repaired report to: {output_path}")


def load_approved_claims(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise TypeError("Expected claim_bank.json to contain a list.")

    return [
        claim for claim in data
        if claim.get("approved_for_report") is True
    ]


def replace_section(
    text: str,
    start_heading: str,
    end_heading: str,
    replacement: str,
) -> str:
    start = text.find(start_heading)
    end = text.find(end_heading)

    if start == -1:
        raise ValueError(f"Start heading not found: {start_heading}")
    if end == -1:
        raise ValueError(f"End heading not found: {end_heading}")
    if end <= start:
        raise ValueError(f"End heading appears before start heading: {end_heading}")

    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def replace_subsection(
    text: str,
    start_heading: str,
    end_heading: str,
    replacement: str,
) -> str:
    return replace_section(text, start_heading, end_heading, replacement)


def build_ownership_section(claims: list[dict[str, Any]]) -> str:
    owner_counts = Counter(claim["owner"] for claim in claims)
    by_owner = defaultdict(list)

    for claim in claims:
        by_owner[claim["owner"]].append(claim)

    lines = [
        "## 1. Ownership and competitive landscape",
        "",
        "**Fact**:",
        f"Henkel contributes 6 patents in this set and Bostik contributes 4 patents. "
        f"The approved claim bank contains {owner_counts.get('Henkel', 0)} Henkel claim items "
        f"and {owner_counts.get('Bostik', 0)} Bostik claim items.",
        "",
        "**Interpretation**:",
        "Henkel's approved claims cover several polymer and formulation routes, while Bostik's approved claims concentrate on olefin block copolymers, single-site catalyst polypropylene systems, and metallocene-catalyzed propylene-based copolymers.",
        "",
        "| Owner | Patent | Directly supported technical signal | Evidence | R&D relevance |",
        "|---|---|---|---|---|",
    ]

    for owner in ["Henkel", "Bostik"]:
        for claim in select_claims(by_owner[owner], claim_type="core_technology"):
            lines.append(
                "| "
                f"{escape_md(owner)} | "
                f"{escape_md(claim['patent_publication'])} | "
                f"{escape_md(claim['claim_text'])} | "
                f"{evidence_refs(claim, max_ids=2)} | "
                f"{escape_md(short_rd_relevance_for_claim(claim, claims))} |"
            )

    return "\n".join(lines)


def build_temporal_section(claims: list[dict[str, Any]]) -> str:
    by_patent = group_by_patent(claims)

    lines = [
        "## 2. Temporal trends",
        "",
        "**Fact**:",
        "The priority window runs from 2009 to 2021. The earlier patents in the set include viscosity-reduction, olefin block copolymer, wax-modified ethylene copolymer, and polar-functional polymer routes; later Bostik patents include single-site catalyst polypropylene and metallocene-catalyzed propylene systems.",
        "",
        "**Interpretation**:",
        "The timeline should be read as a technology-sequence signal inside this limited dataset, not as a market-wide filing trend.",
        "",
        "| Priority year | Patent | Supported technology signal | Evidence | R&D implication |",
        "|---:|---|---|---|---|",
    ]

    for publication in TARGET_PUBLICATIONS:
        patent_claims = by_patent.get(publication, [])
        core = first_claim(patent_claims, "core_technology")
        rd = first_claim(patent_claims, "performance_evidence") or first_claim(
            patent_claims, "rd_relevance"
        )

        if not core:
            continue

        lines.append(
            "| "
            f"{escape_md(core['priority_year'])} | "
            f"{escape_md(publication)} | "
            f"{escape_md(core['claim_text'])} | "
            f"{evidence_refs(core, max_ids=2)} | "
            f"{escape_md(rd['claim_text'] if rd else 'No approved R&D claim available.')} |"
        )

    return "\n".join(lines)


def build_formulation_subsection(claims: list[dict[str, Any]]) -> str:
    formulation_claims = [
        claim for claim in claims
        if claim.get("claim_type") == "formulation_strategy"
    ]

    lines = [
        "### 3.2 Formulation strategies",
        "",
        "| Patent | Direct formulation lever | Evidence | R&D experiment or action |",
        "|---|---|---|---|",
    ]

    for claim in formulation_claims:
        lines.append(
            "| "
            f"{escape_md(claim['patent_publication'])} | "
            f"{escape_md(claim['claim_text'])} | "
            f"{evidence_refs(claim, max_ids=2)} | "
            f"{escape_md(action_from_formulation_claim(claim))} |"
        )

    return "\n".join(lines)


def build_application_subsection(claims: list[dict[str, Any]]) -> str:
    application_claims = [
        claim for claim in claims
        if claim.get("claim_type") == "application_focus"
    ]

    lines = [
        "### 3.3 Application focus",
        "",
        "| Patent | Supported application focus | Evidence | Success metric to test |",
        "|---|---|---|---|",
    ]

    for claim in application_claims:
        lines.append(
            "| "
            f"{escape_md(claim['patent_publication'])} | "
            f"{escape_md(claim['claim_text'])} | "
            f"{evidence_refs(claim, max_ids=2)} | "
            f"{escape_md(success_metric_from_application_claim(claim))} |"
        )

    return "\n".join(lines)


def build_rd_implications_section(claims: list[dict[str, Any]]) -> str:
    rd_claims = [
        claim for claim in claims
        if claim.get("claim_type") in {"performance_evidence", "rd_relevance"}
    ]

    lines = [
        "## 4. Key insights and R&D implications",
        "",
        "| Patent | Action | Evidence behind it | Technical risk to control |",
        "|---|---|---|---|",
    ]

    for claim in rd_claims:
        lines.append(
            "| "
            f"{escape_md(claim['patent_publication'])} | "
            f"{escape_md(action_from_rd_claim(claim))} | "
            f"{evidence_refs(claim, max_ids=2)} | "
            f"{escape_md(risk_from_claim(claim))} |"
        )

    lines.extend(
        [
            "",
            "The immediate technical next step is to run route-specific formulation screens rather than treating the patent set as one homogeneous adhesive family. The first screen should compare polymer route, tackifier/wax package, application temperature, and substrate type using the success metrics above.",
        ]
    )

    return "\n".join(lines)


def group_by_patent(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)

    for claim in claims:
        grouped[claim["patent_publication"]].append(claim)

    return grouped


def select_claims(
    claims: list[dict[str, Any]],
    claim_type: str,
) -> list[dict[str, Any]]:
    return [claim for claim in claims if claim.get("claim_type") == claim_type]


def first_claim(
    claims: list[dict[str, Any]],
    claim_type: str,
) -> dict[str, Any] | None:
    for claim in claims:
        if claim.get("claim_type") == claim_type:
            return claim
    return None


def short_rd_relevance_for_claim(
    claim: dict[str, Any],
    all_claims: list[dict[str, Any]],
) -> str:
    patent = claim["patent_publication"]

    for candidate in all_claims:
        if (
            candidate.get("patent_publication") == patent
            and candidate.get("claim_type") == "performance_evidence"
        ):
            return candidate["claim_text"]

    for candidate in all_claims:
        if (
            candidate.get("patent_publication") == patent
            and candidate.get("claim_type") == "application_focus"
        ):
            return candidate["claim_text"]

    return "Use as a technology checkpoint for route-specific R&D screening."


def action_from_formulation_claim(claim: dict[str, Any]) -> str:
    text = claim["claim_text"].lower()

    if "tackifier" in text:
        return "Screen tackifier chemistry against viscosity, colour, odour, and thermal stability."

    if "wax" in text:
        return "Vary wax type, acid number, and wax/polymer ratio to test processability and substrate wetting."

    if "plasticizer" in text:
        return "Test plasticizer level against flexibility, wetting, and cohesive strength."

    if "resin" in text:
        return "Compare resin families under the same polymer route and coating temperature."

    return "Run a formulation screen using this claim as the controlled design lever."


def success_metric_from_application_claim(claim: dict[str, Any]) -> str:
    text = claim["claim_text"].lower()

    if "elastic" in text or "diaper" in text or "leg cuff" in text:
        return "Creep resistance, peel strength, elastic recovery, and bleed-through."

    if "nonwoven" in text or "film" in text:
        return "Peel strength, coating uniformity, substrate compatibility, and wet-out."

    if "paper" in text or "bookbinding" in text or "packaging" in text:
        return "Open time, bond strength, viscosity, thermal stability, and application temperature."

    return "Bond strength, viscosity, coating quality, and substrate compatibility."


def action_from_rd_claim(claim: dict[str, Any]) -> str:
    text = claim["claim_text"].lower()

    if "viscosity" in text:
        return "Measure melt viscosity across application temperatures and compare coating quality."

    if "creep" in text:
        return "Run elastic-attachment creep testing under strain."

    if "bleed-through" in text or "tan δ" in text:
        return "Use rheology and laminate inspection to screen bleed-through risk."

    if "peel" in text:
        return "Run peel-strength and heat-aging tests."

    if "thermal" in text:
        return "Run thermal-stability and colour/odour checks."

    return "Convert the claim into a route-specific formulation experiment."


def risk_from_claim(claim: dict[str, Any]) -> str:
    text = claim["claim_text"].lower()

    risks = []

    if "viscosity" in text:
        risks.append("viscosity")
    if "creep" in text:
        risks.append("creep")
    if "peel" in text:
        risks.append("peel strength")
    if "thermal" in text or "heat" in text:
        risks.append("thermal stability")
    if "bleed" in text:
        risks.append("bleed-through")
    if "wet" in text:
        risks.append("wet-out")
    if "substrate" in text:
        risks.append("substrate compatibility")
    if "colour" in text or "color" in text:
        risks.append("colour")
    if "odour" in text or "odor" in text:
        risks.append("odour")

    return ", ".join(risks) if risks else "processability and substrate compatibility"


def evidence_refs(claim: dict[str, Any], max_ids: int) -> str:
    evidence_ids = claim.get("evidence_ids", [])

    if not isinstance(evidence_ids, list):
        return "not available"

    return ", ".join(f"[{eid}]" for eid in evidence_ids[:max_ids])


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()

if __name__ == "__main__":
    main()
    