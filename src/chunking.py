import re


def create_chunks(pages, chunk_size=700, overlap=100):
    """
    Create readable, overlapping chunks from extracted PDF pages.

    Strategy:
    1. Split page text into paragraphs.
    2. Combine paragraphs until the chunk reaches the target size.
    3. Keep a small overlap between chunks.
    4. Preserve page number and chunk ID.
    """

    chunks = []

    for page in pages:
        page_number = page["page_number"]
        text = page["text"].strip()

        if not text:
            continue

        # Normalize excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Split into sentence-like units
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        previous_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # If adding the sentence keeps the chunk within the limit
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = (
                    current_chunk + " " + sentence
                ).strip()

            else:
                # Save the current chunk
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "page_number": page_number,
                        "chunk_id": len(chunks) + 1
                    })

                # Create overlap from the end of previous chunk
                overlap_text = current_chunk[-overlap:] if current_chunk else ""

                current_chunk = (
                    overlap_text + " " + sentence
                ).strip()

        # Save remaining text
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "page_number": page_number,
                "chunk_id": len(chunks) + 1
            })

    return chunks