\# PatentLLM Pipeline Design



\## Objective

PatentLLM generates R\&D-focused patent intelligence reports from patent PDFs. The system is designed to reduce hallucination and citation mismatch by moving from raw RAG generation to claim-bank-grounded generation.



\## Pipeline Stages

1\. PDF parsing and section-aware chunking

2\. Embedding and vector indexing

3\. Evidence-pack construction

4\. Refined patent-card generation

5\. Repaired patent-card validation

6\. Atomic claim-bank construction

7\. Claim-bank report generation

8\. Deterministic table repair

9\. Structural audit

10\. Claim-bank alignment audit



\## Why basic RAG was not enough

Initial reports were structurally complete but had weak citation support. Evidence IDs were valid, but many cited rows were too broad or citation-irrelevant.



\## Final accepted architecture

The accepted architecture is:



PDFs → parser/chunker → vector index → evidence pack → refined cards → repaired cards → approved atomic claim bank → claim-bank report → deterministic table repair → alignment audit



\## Evaluation Result

The final claim-bank alignment audit checked 55 report rows:

\- 42 aligned

\- 13 partially aligned

\- 0 citation-irrelevant

\- 0 invalid



This means 55/55 rows were aligned or partially aligned with approved claim-bank items.



\## Final Decision

The final accepted output is the claim-bank + deterministic-table report because it preserves report structure, keeps citations valid, removes citation spam, and validates report rows against approved atomic claims.

