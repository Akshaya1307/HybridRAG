from src.ingestion import extract_text_from_pdf
from src.chunking import create_chunks

pdf_path = "data/documents/sample.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = create_chunks(pages)

print(f"\nTotal chunks: {len(chunks)}\n")

for chunk in chunks:
    print("=" * 80)
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page_number']}")
    print(chunk["text"])