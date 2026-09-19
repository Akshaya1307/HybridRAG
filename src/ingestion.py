import fitz
from pathlib import Path


def extract_text_from_pdf(file_path):
    """
    Extract text from every page of a PDF.

    Returns:
        list: A list of dictionaries containing
              page number and extracted text.
    """

    document = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

    document.close()

    return pages