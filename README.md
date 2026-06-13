# PatentLLM Step 1: Patent Parsing and Chunking

This is the first stage of the PatentLLM system. It converts raw patent PDFs into structured JSONL files for downstream embedding, indexing, retrieval, LLM report generation, and evaluation.

## What this stage does

- Extracts native PDF text when available.
- Falls back to OCR for scanned/image PDFs.
- Preserves page-level traceability.
- Extracts lightweight patent metadata.
- Detects patent sections.
- Builds paragraph records.
- Creates RAG-ready chunks.
- Writes a document-level quality report.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
tesseract --version
```

## Run

Put patent PDFs in:

```text
data/raw/patents/
```

Then run:

```bash
python scripts/parse_patents.py \
  --input-dir data/raw/patents \
  --output-dir data/processed/parsed_patents \
  --ocr-dpi 240
```

For a quick smoke test:

```bash
python scripts/parse_patents.py \
  --input-dir data/raw/patents \
  --output-dir data/processed/parsed_patents \
  --max-pages 2 \
  --ocr-dpi 180
```

## Outputs

```text
data/processed/parsed_patents/documents.jsonl
data/processed/parsed_patents/pages.jsonl
data/processed/parsed_patents/paragraphs.jsonl
data/processed/parsed_patents/chunks.jsonl
data/processed/parsed_patents/quality_report.jsonl
```

Embed `chunks.jsonl` in the next stage. Review `quality_report.jsonl` before indexing.
