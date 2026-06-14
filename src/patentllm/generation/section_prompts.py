"""Section-wise prompt building for detailed R&D patent reports.

This module solves the main limitation of single-pass local LLM generation:
a full patent intelligence report can exceed the output budget of small local
models. Instead, each report section is generated separately and then assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SYSTEM_PROMPT = """You are a senior patent strategy and R&D technology intelligence analyst.

You write evidence-based patent analysis reports for R&D teams.

Rules:
- Use only the supplied evidence.
- Do not invent owners, dates, measurements, patent relationships, examples, or technical effects.
- Every factual or technical paragraph must cite evidence IDs like [E4], [E12].
- Do not make legal freedom-to-operate, infringement, validity, clearance, opposition, or patentability conclusions.
- If evidence is incomplete, say so clearly.
- Write in concise, professional R&D intelligence style.
- Output only the requested section.
"""


@dataclass(frozen=True, slots=True)
class SectionSpec:
    """One report section generation unit."""

    section_id: str
    title: str
    instruction: str
    preferred_categories: tuple[str, ...]
    max_items: int
    max_chars_per_item: int
    max_new_tokens: int


SECTION_SPECS: list[SectionSpec] = [
    SectionSpec(
        section_id="scope_summary",
        title="Opening sections",
        preferred_categories=("claim", "polymer", "application", "formulation"),
        max_items=24,
        max_chars_per_item=380,
        max_new_tokens=1200,
        instruction="""Generate exactly these sections:

# Patent Analysis for R&D
Technology intelligence brief on selected hot-melt adhesive patent publications

## Scope snapshot
Write exactly four bullets:
- Scope:
- Owners:
- Priority window:
- Main technology cluster:

## Main conclusion
Write one analytical paragraph of 3-5 sentences.

## Executive summary
Write 3 concise paragraphs:
1. Patent-set coverage.
2. Ownership and technical movement.
3. R&D next step.

## Scope and evidence base
Write three short paragraphs:
- Fact.
- Method.
- Interpretation.

Use owner counts and priority years from metadata. Cite evidence IDs throughout.""",
    ),
    SectionSpec(
        section_id="ownership",
        title="Ownership and competitive landscape",
        preferred_categories=("claim", "polymer", "formulation", "application"),
        max_items=26,
        max_chars_per_item=360,
        max_new_tokens=1000,
        instruction="""Generate exactly this section:

## 1. Ownership and competitive landscape

Write:
- one Fact paragraph
- one Interpretation paragraph
- one markdown table with columns:
  Owner signal | Evidence in this set | What it means | R&D relevance

Use owner counts from metadata.
Do not make legal or FTO statements.
Cite evidence IDs throughout.""",
    ),
    SectionSpec(
        section_id="temporal",
        title="Temporal trends",
        preferred_categories=("claim", "polymer", "formulation", "application"),
        max_items=26,
        max_chars_per_item=360,
        max_new_tokens=1000,
        instruction="""Generate exactly this section:

## 2. Temporal trends

Write:
- one Fact paragraph
- one Interpretation paragraph
- one markdown table with columns:
  Time signal | Evidence | Interpretation | R&D implication

Use the supplied priority years only.
Do not invent filing waves beyond the evidence.
Cite evidence IDs throughout.""",
    ),
    SectionSpec(
        section_id="technology_deep_dive",
        title="Technology deep dive and R&D relevance",
        preferred_categories=("claim", "polymer", "formulation", "application", "performance"),
        max_items=38,
        max_chars_per_item=420,
        max_new_tokens=2200,
        instruction="""Generate exactly this section:

## 3. Technology deep dive and R&D relevance

### 3.1 Polymer types
Group patents into chemistry routes. For each route, include:
- route name
- supporting patents
- what the route technically changes
- R&D relevance

### 3.2 Formulation strategies
Create a markdown table:
Repeated problem | Design lever | Evidence patents | R&D experiment or action

### 3.3 Application focus
Explain application areas and success metrics.
Discuss only application areas supported by evidence, such as elastic attachment, stretch laminates, nonwoven/film bonding, paper products, coating quality, wet-state use, or disposable hygiene articles.

Cite evidence IDs throughout.""",
    ),
    SectionSpec(
        section_id="rd_implications",
        title="Key insights and R&D implications",
        preferred_categories=("application", "performance", "formulation", "polymer", "claim"),
        max_items=34,
        max_chars_per_item=420,
        max_new_tokens=1400,
        instruction="""Generate exactly this section:

## 4. Key insights and R&D implications

Create a markdown table:
Action | Evidence behind it | Technical risk to control

The risk column must only discuss technical R&D risks:
viscosity, odour, colour, creep, peel strength, wet adhesion, substrate compatibility, processability, thermal stability, coating quality, formulation stability, or sensory profile.

After the table, write one paragraph identifying the immediate technical next step.

Do not mention infringement, FTO, validity, patentability, clearance, opposition, or legal risk.
Cite evidence IDs throughout.""",
    ),
    SectionSpec(
        section_id="limits_key",
        title="Limits and patent key",
        preferred_categories=("claim", "polymer", "formulation", "application", "performance"),
        max_items=32,
        max_chars_per_item=320,
        max_new_tokens=1600,
        instruction="""Generate exactly these sections:

## 5. Limits of the conclusion
Write 2 concise paragraphs:
- dataset and OCR/evidence-pack limitations
- not legal advice / FTO limitation

## Patent key and extracted R&D signal
Create a markdown table:
# | Patent | Owner | Priority | Core technology | Main R&D relevance

Include all 10 target patent publications:
EP2756049A1, EP2841522A1, EP2411423A1, EP2958970A1, EP2686395A2, EP3268442A1, EP2456819A1, EP3402857A1, EP3707219A1, EP4441161A1.

Cite evidence IDs where technical descriptions are used.""",
    ),
]


def build_section_prompt(
    evidence_pack: dict[str, Any],
    section_spec: SectionSpec,
) -> str:
    """Build one section-specific prompt."""

    selected_items = select_section_evidence(
        evidence_items=evidence_pack["evidence_items"],
        preferred_categories=section_spec.preferred_categories,
        max_items=section_spec.max_items,
    )

    evidence_block = "\n\n".join(
        format_evidence_item(
            item=item,
            max_chars=section_spec.max_chars_per_item,
        )
        for item in selected_items
    )

    metadata = evidence_pack["metadata_summary"]
    coverage_block = format_coverage_table(evidence_pack["coverage"])

    return f"""<task>
Generate one section of a patent-based R&D technology intelligence report.

Section:
{section_spec.title}

Audience:
R&D stakeholders.

Objective:
Analyze the supplied patent publications and derive actionable insights relevant for R&D and innovation strategy.
</task>

<metadata>
Documents represented: {metadata.get("documents_represented")}
Owner counts: {metadata.get("owner_counts")}
Priority year range: {metadata.get("priority_year_min")}–{metadata.get("priority_year_max")}
Evidence items used in this section: {len(selected_items)}
</metadata>

<coverage_table>
{coverage_block}
</coverage_table>

<section_instruction>
{section_spec.instruction}
</section_instruction>

<evidence>
{evidence_block}
</evidence>

<final_instruction>
Generate the requested section now.
Use markdown.
Use evidence citations.
Do not output any other section.
</final_instruction>
"""


def select_section_evidence(
    evidence_items: list[dict[str, Any]],
    preferred_categories: tuple[str, ...],
    max_items: int,
) -> list[dict[str, Any]]:
    """Select evidence balanced by patent and section category."""

    by_doc: dict[str, list[dict[str, Any]]] = {}

    for item in evidence_items:
        document_id = str(item.get("document_id", ""))
        by_doc.setdefault(document_id, []).append(item)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    # First: round-robin by document and preferred category.
    for document_id in sorted(by_doc):
        for category in preferred_categories:
            item = first_item_for_category(by_doc[document_id], category)
            append_if_new(selected, seen, item, max_items)

    # Second: fill remaining budget with vector evidence and any remaining useful evidence.
    fallback_categories = (
        "vector_implications",
        "vector_performance",
        "vector_application",
        "vector_formulation",
        "vector_polymer",
        "performance",
        "application",
        "formulation",
        "polymer",
        "claim",
    )

    for category in fallback_categories:
        for document_id in sorted(by_doc):
            item = first_item_for_category(by_doc[document_id], category)
            append_if_new(selected, seen, item, max_items)

    return selected[:max_items]


def first_item_for_category(
    items: list[dict[str, Any]],
    category: str,
) -> dict[str, Any] | None:
    """Return the best item for one category."""

    matches = [
        item for item in items
        if str(item.get("selection_category", "")) == category
    ]

    if not matches:
        return None

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        text_len = len(str(item.get("text", "")))
        evidence_number = evidence_id_number(str(item.get("evidence_id", "")))
        return (-text_len, evidence_number)

    return sorted(matches, key=sort_key)[0]


def append_if_new(
    selected: list[dict[str, Any]],
    seen: set[str],
    item: dict[str, Any] | None,
    max_items: int,
) -> None:
    """Append an evidence item if not already selected."""

    if item is None:
        return

    if len(selected) >= max_items:
        return

    evidence_id = str(item.get("evidence_id", ""))

    if evidence_id in seen:
        return

    selected.append(item)
    seen.add(evidence_id)


def format_evidence_item(item: dict[str, Any], max_chars: int) -> str:
    """Format evidence item compactly."""

    text = clean_text(str(item.get("text", "")))
    text = clip_text(text, max_chars=max_chars)

    return f"""[{item.get("evidence_id", "")}]
patent: {item.get("patent_publication", "not available")}
owner: {item.get("owner", "not available")}
priority: {item.get("priority_year", "not available")}
category: {item.get("selection_category", "")}
section: {item.get("section", "")}
text: {text}
"""


def format_coverage_table(rows: list[dict[str, Any]]) -> str:
    """Format compact coverage table."""

    lines = [
        "patent | owner | priority | claim | polymer | formulation | application | performance"
    ]

    for row in rows:
        lines.append(
            f"{row.get('patent_publication', '')} | "
            f"{row.get('owner', '')} | "
            f"{row.get('priority_year', '')} | "
            f"{row.get('has_claim', '')} | "
            f"{row.get('has_polymer', '')} | "
            f"{row.get('has_formulation', '')} | "
            f"{row.get('has_application', '')} | "
            f"{row.get('has_performance', '')}"
        )

    return "\n".join(lines)


def evidence_id_number(evidence_id: str) -> int:
    """Convert E12 to 12."""

    try:
        return int(evidence_id.upper().replace("E", "").strip())
    except ValueError:
        return 9999


def clean_text(text: str) -> str:
    """Normalize whitespace."""

    return " ".join(text.split())


def clip_text(text: str, max_chars: int) -> str:
    """Clip without cutting mid-word."""

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + " [TRUNCATED]"