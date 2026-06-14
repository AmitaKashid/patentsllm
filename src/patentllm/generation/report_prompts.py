"""Prompt templates for detailed patent R&D report generation.

This module consumes a structured evidence pack and creates a controlled,
evidence-grounded prompt for LLM report generation.

The design goal is not creativity. The design goal is:
- broad coverage across all 10 patents
- evidence-backed technical synthesis
- citation discipline
- business-readable R&D implications
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SYSTEM_PROMPT = """You are a senior patent strategy and R&D technology intelligence analyst.

You write evidence-based patent analysis reports for R&D teams. You translate patent publications into technical themes, formulation variables, competitive signals, and practical R&D implications.

Mandatory rules:
- Use only the supplied evidence pack.
- Do not invent patent owners, dates, citation counts, family relationships, examples, measurements, or technical effects.
- Do not make legal freedom-to-operate, infringement, validity, opposition, clearance, or patentability conclusions.
- Separate facts from interpretation.
- Every factual or technical paragraph must cite evidence IDs such as [E4], [E12].
- If evidence is incomplete, write "not available in the supplied evidence" or "the supplied evidence is insufficient".
- Do not cite evidence IDs that are not in the evidence pack.
- Write in a concise, professional style for senior R&D stakeholders.
- Avoid generic marketing language.
- Output only the final report, not your reasoning process.
"""


@dataclass(frozen=True, slots=True)
class PromptBuildConfig:
    """Controls prompt size and evidence formatting."""

    max_chars_per_evidence: int = 1800
    max_total_evidence_items: int = 63
    include_coverage_table: bool = True


def build_detailed_rd_report_prompt(
    evidence_pack: dict[str, Any],
    config: PromptBuildConfig | None = None,
) -> str:
    """Build a detailed R&D report prompt from an evidence pack JSON object."""

    prompt_config = config or PromptBuildConfig()

    task = evidence_pack["task"]
    metadata_summary = evidence_pack["metadata_summary"]
    coverage_rows = evidence_pack["coverage"]
    evidence_items = evidence_pack["evidence_items"][: prompt_config.max_total_evidence_items]

    coverage_block = (
        _format_coverage_table(coverage_rows)
        if prompt_config.include_coverage_table
        else "Coverage table intentionally omitted."
    )

    evidence_block = "\n\n".join(
        _format_evidence_item(
            item=item,
            max_chars=prompt_config.max_chars_per_evidence,
        )
        for item in evidence_items
    )

    return f"""<task>
Title:
{task.get("title", "")}

Role:
{task.get("role", "")}

Audience:
{task.get("audience", "")}

Objective:
{task.get("objective", "")}

Target patent publications:
{", ".join(task.get("target_publications", []))}

Required sections:
{_format_list(task.get("required_sections", []))}

Report style:
{task.get("report_style", "")}

Notes:
{task.get("notes", "")}
</task>

<evidence_pack_summary>
Pack ID: {evidence_pack.get("pack_id", "")}
Documents represented: {metadata_summary.get("documents_represented", "not available")}
Owner counts from structured metadata: {metadata_summary.get("owner_counts", "not available")}
Priority year minimum: {metadata_summary.get("priority_year_min", "not available")}
Priority year maximum: {metadata_summary.get("priority_year_max", "not available")}
Evidence items provided: {len(evidence_items)}
</evidence_pack_summary>

<coverage_table>
{coverage_block}
</coverage_table>

<reporting_policy>
Use the structured metadata for ownership and timeline signals when available.
Use evidence text for technical interpretation.
Do not treat OCR noise as a fact.
If a chunk is unclear, use it cautiously or state uncertainty.
Do not overclaim based on one patent.
Do not write legal advice.
Do not use external knowledge.
</reporting_policy>

<internal_analysis_plan>
Before writing the final report, internally perform this analysis:
1. Check whether the 10 target patent publications are represented.
2. Map patents to owners and priority years using metadata.
3. Identify major technology routes:
   - EPR / EPDM / semi-crystalline olefin systems
   - olefin block copolymer systems
   - ethylene copolymer plus modified wax systems
   - metallocene or propylene-based polyolefin systems
   - polar-modified / polyester-compatible systems
   - functionalized polyolefin or wet-state systems
4. Identify formulation levers:
   - polymer architecture
   - tackifier type
   - wax package
   - plasticizer/oil
   - stabilizer/antioxidant
   - viscosity and application temperature
5. Identify application targets:
   - nonwoven bonding
   - PE film / nonwoven bonding
   - elastic attachment
   - stretch laminate
   - construction bonding
   - paper/bookbinding/packaging
   - wet-state absorbent article use
6. Translate findings into R&D experiment directions and risk controls.
Do not output this internal analysis.
</internal_analysis_plan>

<required_output>
# Patent Analysis for R&D
Technology intelligence brief on selected hot-melt adhesive patent publications

## Scope snapshot
Write exactly four bullets:
- Scope:
- Owners:
- Priority window:
- Main technology cluster:

## Main conclusion
Write one strong analytical paragraph of 3-5 sentences. Cite evidence IDs.

## Executive summary
Write 3-4 concise paragraphs:
1. What the patent set covers.
2. Ownership and competitive signal.
3. Technical movement across the dataset.
4. What R&D should do with the findings.
Cite evidence IDs in every paragraph.

## Scope and evidence base
Write three short paragraphs:
- Fact.
- Method.
- Interpretation.

## 1. Ownership and competitive landscape
Write:
- Fact paragraph.
- Interpretation paragraph.
- Markdown table with columns:
  Owner signal | Evidence in this set | What it means | R&D relevance

Use owner counts from metadata. Cite evidence or metadata summary where relevant.

## 2. Temporal trends
Write:
- Fact paragraph.
- Interpretation paragraph.
- Markdown table with columns:
  Time signal | Evidence | Interpretation | R&D implication

Use priority years from metadata. Do not invent filing waves beyond the supplied evidence.

## 3. Technology deep dive and R&D relevance

### 3.1 Polymer types
Group the patents into polymer or chemistry routes.
For each route, explain:
- what the chemistry route is
- which patents support it
- what technical problem it appears to address
- why it matters for R&D
Cite evidence IDs.

### 3.2 Formulation strategies
Start with one short paragraph explaining the formulation logic.

Then create a markdown table with columns:
Repeated problem | Design lever | Evidence patents | R&D experiment or action

Only include rows supported by evidence.

### 3.3 Application focus
Explain the application areas and their different success metrics.
Distinguish application types only when supported by evidence.

## 4. Key insights and R&D implications
Create a markdown table with columns:
Action | Evidence behind it | Risk to control

Then write one paragraph identifying the immediate technical next step for R&D.

## 5. Limits of the conclusion
Write 2 concise paragraphs:
- dataset and OCR/evidence-pack limitations
- not legal advice / FTO limitation

## Patent key and extracted R&D signal
Create a markdown table with columns:
# | Patent | Owner | Priority | Core technology | Main R&D relevance
</required_output>

<evidence>
{evidence_block}
</evidence>

<final_instruction>
Generate the final report now.
Use the exact section order.
Use markdown.
Cite evidence IDs throughout.
Do not add extra sections.
Do not mention that you are an AI model.
</final_instruction>
"""


def _format_evidence_item(item: dict[str, Any], max_chars: int) -> str:
    text = _clean_text(str(item.get("text", "")))
    clipped_text = _clip_text(text, max_chars=max_chars)

    page_start = item.get("page_start")
    page_end = item.get("page_end")
    page_range = (
        f"{page_start}-{page_end}"
        if page_start is not None and page_end is not None
        else "unknown"
    )

    return f"""[{item.get("evidence_id", "")}]
document_id: {item.get("document_id", "")}
patent_publication: {item.get("patent_publication", "not available")}
owner: {item.get("owner", "not available")}
priority_year: {item.get("priority_year", "not available")}
chunk_id: {item.get("chunk_id", "")}
chunk_type: {item.get("chunk_type", "")}
section: {item.get("section", "")}
pages: {page_range}
selection_category: {item.get("selection_category", "")}
selection_reason: {item.get("selection_reason", "")}
text:
{clipped_text}
"""


def _format_coverage_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "document_id | patent | owner | priority | items | claim | polymer | "
        "formulation | application | performance"
    )

    lines = [header]

    for row in rows:
        lines.append(
            f"{row.get('document_id', '')} | "
            f"{row.get('patent_publication', '')} | "
            f"{row.get('owner', '')} | "
            f"{row.get('priority_year', '')} | "
            f"{row.get('selected_items', '')} | "
            f"{row.get('has_claim', '')} | "
            f"{row.get('has_polymer', '')} | "
            f"{row.get('has_formulation', '')} | "
            f"{row.get('has_application', '')} | "
            f"{row.get('has_performance', '')}"
        )

    return "\n".join(lines)


def _format_list(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rsplit(" ", 1)[0]
    return f"{clipped}\n[TRUNCATED_FOR_PROMPT_BUDGET]"