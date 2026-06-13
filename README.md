# PatentLLM — Step 1 Parser v4

Patent-aware parsing and chunking for scanned patent PDFs.

## What v4 fixes

- OCR-aware PDF text extraction using PyMuPDF/Tesseract.
- Safer metadata extraction: metadata is retained for display/filtering but is not treated as primary evidence.
- Robust claim extraction for OCR-heavy PCT patents:
  - ignores early cover-page references to international search reports;
  - trims real search-report pages only after the claim block begins;
  - keeps short claim fragments that previous paragraph filtering could drop;
  - infers low-confidence claim 1 when OCR drops the short `1.` line;
  - flags missing or low-confidence claims instead of silently embedding bad legal evidence.
- Quality reports include embeddable chunk counts, detected claim numbers, missing claim numbers, and warnings.

## Run

```powershell
python scripts/parse_patents.py `
  --input-dir data/raw/patents `
  --output-dir data/processed/parsed_patents `
  --ocr-dpi 240 `
  --disable-ocr-confidence

python scripts/audit_parser_outputs.py
```

## Embed only safe chunks

Stage 2 indexing should embed only records where:

```python
chunk["embeddable"] is True
```

Do not embed metadata chunks as technical evidence.
