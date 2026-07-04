import fitz


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be parsed by the extraction library."""


def extract_pdf_text(file_bytes: bytes, filename: str, max_pages: int | None = None) -> dict:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError("Unable to open PDF file") from exc

    pages = []
    full_text_parts = []

    try:
        if max_pages is not None and document.page_count > max_pages:
            raise PDFExtractionError(f"PDF has {document.page_count} pages, which exceeds the {max_pages}-page limit")

        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            pages.append(
                {
                    "page_number": page_index,
                    "text": text,
                    "char_count": len(text),
                }
            )
            if text:
                full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)

        return {
            "filename": filename,
            "page_count": document.page_count,
            "char_count": len(full_text),
            "pages": pages,
            "full_text": full_text,
            "library_used": "PyMuPDF",
        }
    finally:
        document.close()
