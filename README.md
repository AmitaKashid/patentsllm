# \# PatentLLM

# 

# PatentLLM is an end-to-end patent intelligence pipeline for generating R\&D-focused reports from patent PDF documents. The project focuses on patent parsing, section-aware chunking, embedding-based retrieval, evidence-pack construction, local LLM report generation, and citation-grounded evaluation.

# 

# The main engineering goal is not only to generate a readable patent report, but to reduce unsupported claims and citation mismatch in LLM-generated patent analysis.

# 

# \## Problem

# 

# Basic RAG report generation can produce structurally good reports while still attaching citations to claims that are too broad, weakly supported, or not directly grounded in the cited evidence.

# 

# Patent reports are especially sensitive to this problem because technical claims, formulation details, application areas, and R\&D implications need to remain traceable to patent evidence.

# 

# PatentLLM addresses this by moving beyond direct RAG generation and introducing a controlled grounding pipeline.

# 

# \## Final Accepted Architecture

# 

# The final accepted architecture is:

# 

# ```text

# Patent PDFs

# → parser and section-aware chunker

# → vector index

# → evidence pack

# → refined patent cards

# → repaired patent cards

# → approved atomic claim bank

# → claim-bank report generation

# → deterministic table repair

# → structural audit

# → claim-bank alignment audit

# ```

# 

# \## Pipeline Stages

# 

# \### 1. PDF Parsing and Chunking

# 

# Patent PDFs are parsed into structured outputs:

# 

# ```text

# documents.jsonl

# pages.jsonl

# paragraphs.jsonl

# chunks.jsonl

# quality\_report.jsonl

# ```

# 

# The parser extracts document-level metadata, page text, paragraphs, claims, description sections, examples, and chunk-level records for downstream retrieval.

# 

# \### 2. Embedding and Indexing

# 

# Parsed chunks are embedded and indexed for semantic retrieval. The project supports local open-source embedding models, including:

# 

# ```text

# BAAI/bge-m3

# AI-Growth-Lab/PatentSBERTa

# ```

# 

# The index is stored in Chroma and used to retrieve patent evidence for report generation.

# 

# \### 3. Evidence Pack Construction

# 

# The evidence-pack stage selects relevant evidence for the R\&D use case. It filters out noisy metadata/search-report chunks and keeps patent-specific evidence for:

# 

# ```text

# claims

# polymer routes

# formulation strategies

# application focus

# performance evidence

# R\&D relevance

# ```

# 

# \### 4. Patent Cards

# 

# A patent-card layer is created to avoid forcing the LLM to reason directly from long raw chunks. Each patent card summarizes one patent using grounded fields:

# 

# ```text

# patent\_publication

# owner

# priority\_year

# core\_technology

# polymer\_route

# formulation\_strategy

# application\_focus

# performance\_evidence

# rd\_relevance

# supporting\_evidence\_ids

# field\_evidence\_map

# ```

# 

# \### 5. Repaired Patent Cards

# 

# A repair and validation pass ensures that every card has valid supporting evidence IDs and field-level evidence mappings before report generation.

# 

# \### 6. Atomic Claim Bank

# 

# The atomic claim bank converts repaired patent cards into small approved claim/evidence pairs. This is the main grounding mechanism.

# 

# Each claim contains:

# 

# ```text

# claim\_id

# patent\_publication

# owner

# priority\_year

# claim\_type

# report\_section

# claim\_text

# evidence\_ids

# approved\_for\_report

# rejection\_reason

# ```

# 

# Generic or weak R\&D claims are rejected before report generation.

# 

# \### 7. Claim-Bank Report Generation

# 

# The report is generated from approved atomic claims, not directly from raw retrieved chunks. This reduces citation mismatch because the LLM can only cite evidence IDs attached to approved claims.

# 

# \### 8. Deterministic Table Repair

# 

# High-risk report tables are repaired deterministically from the claim bank. This prevents broad multi-citation table rows and keeps each row tied to approved patent claims.

# 

# \### 9. Evaluation and Auditing

# 

# The project includes several audit layers:

# 

# ```text

# structural report audit

# body citation audit

# citation validity audit

# citation-support audit

# claim-bank alignment audit

# ```

# 

# The final accepted report passed the structural audit and claim-bank alignment audit.

# 

# \## Final Evaluation Result

# 

# The final claim-bank alignment audit checked 55 report rows.

# 

# ```text

# Aligned: 42

# Partially aligned: 13

# Aligned or partially aligned: 55/55

# ```

# 

# This means every checked row in the final deterministic-table report was aligned or partially aligned with approved claim-bank items.

# 

# \## Final Accepted Output

# 

# The accepted report is:

# 

# ```text

# docs/results/final\_claim\_bank\_grounded\_report.md

# ```

# 

# The accepted architecture is:

# 

# ```text

# claim-bank + deterministic tables

# ```

# 

# This architecture was selected because it:

# 

# \* preserves the full R\&D report structure

# \* keeps all citation IDs valid

# \* removes citation-spam rows

# \* grounds report rows in approved atomic claims

# \* produces a reproducible audit trail

# 

# \## Current Project Status

# 

# Implemented:

# 

# \* patent PDF parser

# \* section-aware chunking

# \* Chroma vector indexing

# \* local embedding model support

# \* evidence-pack generation

# \* local LLM report generation with Ollama

# \* patent-card generation and repair

# \* atomic claim-bank construction

# \* claim-bank report generation

# \* deterministic table repair

# \* structural and alignment audits

# \* GitHub-ready documentation structure

# 

# \## Repository Structure

# 

# ```text

# PatentLLM/

# ├── configs/

# ├── data/

# │   ├── raw/

# │   ├── processed/

# │   └── reference/

# ├── docs/

# │   ├── architecture/

# │   └── results/

# ├── scripts/

# ├── src/

# │   └── patentllm/

# ├── tests/

# ├── README.md

# └── pyproject.toml

# ```

# 

# \## Running the Pipeline

# 

# \### Parse patent PDFs

# 

# ```powershell

# python scripts/parse\_patents.py `

# &#x20; --input-dir data/raw/patents `

# &#x20; --output-dir data/processed/parsed\_patents `

# &#x20; --ocr-dpi 240 `

# &#x20; --disable-ocr-confidence

# ```

# 

# \### Build vector index

# 

# ```powershell

# python scripts/build\_index.py `

# &#x20; --chunks-path data/processed/parsed\_patents/chunks.jsonl `

# &#x20; --persist-dir data/processed/vector\_store/chroma `

# &#x20; --collection-name patent\_chunks\_bge\_m3 `

# &#x20; --embedding-backend sentence\_transformer `

# &#x20; --model-name BAAI/bge-m3 `

# &#x20; --batch-size 8

# ```

# 

# \### Build evidence pack

# 

# ```powershell

# python scripts/build\_report\_evidence\_pack.py `

# &#x20; --task-config configs/report\_tasks/henkel\_rd\_use\_case.json `

# &#x20; --chunks-path data/processed/parsed\_patents/chunks.jsonl `

# &#x20; --metadata-csv data/reference/patent\_metadata.csv `

# &#x20; --collection-name patent\_chunks\_bge\_m3 `

# &#x20; --embedding-model-name BAAI/bge-m3 `

# &#x20; --max-items-per-doc 6 `

# &#x20; --max-vector-items-per-section 5 `

# &#x20; --max-total-items 70

# ```

# 

# \### Generate refined patent cards

# 

# ```powershell

# python scripts/refine\_patent\_cards\_with\_llm.py `

# &#x20; --evidence-pack data/processed/evidence\_packs/<RUN\_ID>/evidence\_pack.json `

# &#x20; --model qwen3:8b `

# &#x20; --temperature 0.0 `

# &#x20; --num-ctx 12000 `

# &#x20; --num-predict 1800

# ```

# 

# \### Repair refined cards

# 

# ```powershell

# python scripts/repair\_refined\_patent\_cards.py `

# &#x20; --cards-json data/processed/patent\_cards\_refined/<RUN\_ID>/patent\_cards\_refined.json `

# &#x20; --evidence-pack data/processed/evidence\_packs/<RUN\_ID>/evidence\_pack.json

# ```

# 

# \### Build claim bank

# 

# ```powershell

# python scripts/build\_claim\_bank\_from\_refined\_cards.py `

# &#x20; --cards-json data/processed/patent\_cards\_refined\_repaired/<RUN\_ID>/patent\_cards\_refined\_repaired.json

# ```

# 

# \### Generate claim-bank report

# 

# ```powershell

# python scripts/generate\_report\_from\_claim\_bank.py `

# &#x20; --claim-bank-json data/processed/claim\_banks/<RUN\_ID>/claim\_bank.json `

# &#x20; --llm ollama:qwen3:8b `

# &#x20; --temperature 0.0 `

# &#x20; --num-ctx 12000 `

# &#x20; --num-predict 4200

# ```

# 

# \### Repair high-risk tables deterministically

# 

# ```powershell

# python scripts/repair\_claim\_bank\_report\_tables.py `

# &#x20; --report-path data/processed/generated\_reports\_from\_claim\_bank/<RUN\_ID>/final\_report\_ollama\_qwen3\_8b.md `

# &#x20; --claim-bank-json data/processed/claim\_banks/<RUN\_ID>/claim\_bank.json

# ```

# 

# \### Run structural audit

# 

# ```powershell

# python scripts/audit\_generated\_report.py `

# &#x20; --report-path data/processed/generated\_reports\_from\_claim\_bank/<RUN\_ID>/final\_report\_ollama\_qwen3\_8b\_deterministic\_tables.md `

# &#x20; --evidence-pack data/processed/evidence\_packs/<RUN\_ID>/evidence\_pack.json

# ```

# 

# \### Run claim-bank alignment audit

# 

# ```powershell

# python scripts/audit\_claim\_bank\_report\_alignment.py `

# &#x20; --report-path data/processed/generated\_reports\_from\_claim\_bank/<RUN\_ID>/final\_report\_ollama\_qwen3\_8b\_deterministic\_tables.md `

# &#x20; --claim-bank-json data/processed/claim\_banks/<RUN\_ID>/claim\_bank.json

# ```

# 

# \## Models Used

# 

# Local generation was tested with Ollama models including:

# 

# ```text

# qwen3:8b

# llama3.1:8b

# mistral:7b

# ```

# 

# The final accepted claim-bank report was generated with:

# 

# ```text

# qwen3:8b

# ```

# 

# \## Key Learning

# 

# Direct RAG was not enough for reliable patent report generation. The strongest architecture used a layered grounding strategy:

# 

# ```text

# retrieved evidence

# → patent cards

# → repaired cards

# → approved atomic claims

# → deterministic high-risk tables

# ```

# 

# This made the report more auditable, reproducible, and suitable for technical R\&D decision support.

# 

# \## Limitations

# 

# This project does not perform legal analysis, infringement analysis, validity analysis, patentability analysis, opposition analysis, clearance analysis, or freedom-to-operate analysis.

# 

# The generated report is intended for technical R\&D intelligence, not legal decision-making.

# 

# \## License

# 

# Add your chosen license here.



