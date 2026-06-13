"""PDF text extraction with OCR fallback and optional OCR quality signals."""

from __future__ import annotations

import os
from pathlib import Path

import fitz  # PyMuPDF

from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import PageRecord
from patentllm.parsing.text_cleaning import normalize_page_text


class PatentPdfTextExtractor:
    """Extract page text from patent PDFs with native extraction first and OCR fallback."""

    def __init__(self, config: PatentParserConfig) -> None:
        self.config = config
        self.tessdata_dir = self._resolve_tessdata_dir(config.tessdata_dir)

        if self.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)

    def extract_pages(self, pdf_path: Path, document_id: str) -> list[PageRecord]:
        """Extract text page-by-page from one PDF."""

        source_file = pdf_path.name
        pages: list[PageRecord] = []

        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
            limit = min(page_count, self.config.max_pages) if self.config.max_pages else page_count

            for page_index in range(limit):
                page = doc[page_index]
                text, method = self._extract_text(page)

                pages.append(
                    PageRecord(
                        document_id=document_id,
                        source_file=source_file,
                        page_number=page_index + 1,
                        text=text,
                        extraction_method=method,
                        char_count=len(text),
                    )
                )

        return pages

    def _extract_text(self, page: fitz.Page) -> tuple[str, str]:
        """Extract native text first; use OCR only when native text is insufficient."""

        native_text = normalize_page_text(page.get_text("text", sort=True))

        if len(native_text) >= self.config.min_native_chars_per_page:
            return native_text, "native_pdf_text"

        if not self.config.ocr_enabled:
            return native_text, "native_pdf_text_short"

        try:
            textpage = page.get_textpage_ocr(
                language=self.config.ocr_language,
                dpi=self.config.ocr_dpi,
                full=True,
                tessdata=str(self.tessdata_dir) if self.tessdata_dir else None,
            )
            ocr_text = normalize_page_text(
                page.get_text("text", sort=True, textpage=textpage)
            )
        except Exception as exc:
            raise RuntimeError(
                "OCR was required but failed. Check that Tesseract is installed and "
                "that the tessdata directory contains eng.traineddata. You can pass it with "
                "--tessdata-dir, for example: "
                "--tessdata-dir \"C:\\Program Files\\Tesseract-OCR\\tessdata\""
            ) from exc

        if len(ocr_text) > len(native_text):
            return ocr_text, "tesseract_ocr"

        return native_text, "native_pdf_text_short"

    @staticmethod
    def _resolve_tessdata_dir(explicit_path: str | None) -> Path | None:
        """Resolve the Tesseract tessdata directory across common Windows installs."""

        candidates: list[Path] = []

        if explicit_path:
            candidates.append(Path(explicit_path))

        env_path = os.environ.get("TESSDATA_PREFIX")
        if env_path:
            candidates.append(Path(env_path))

        candidates.extend(
            [
                Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
            ]
        )

        for candidate in candidates:
            if (candidate / "eng.traineddata").exists():
                return candidate

        return None