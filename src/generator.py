import ollama


MODEL_NAME = "qwen3:1.7b"


def generate_answer(question, retrieved_results):
    """
    Generate a concise answer using Qwen3
    from retrieved document chunks.

    Supports both:
    1. Normal chunks from test_qwen.py
    2. HybridRAG results from app.py
    """

    context_parts = []

    for result in retrieved_results:

        # HybridRAG result:
        # {"chunk": {...}, "rrf_score": ...}
        if "chunk" in result:
            chunk = result["chunk"]

        # Normal chunk:
        # {"page_number": ..., "text": ...}
        else:
            chunk = result

        page_number = chunk.get("page_number", "Unknown")
        chunk_id = chunk.get("chunk_id", "N/A")
        text = chunk.get("text", "")

        context_parts.append(
            f"Source: Page {page_number}\n"
            f"Chunk ID: {chunk_id}\n"
            f"{text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the document context below.

Rules:
- Use only the provided context.
- Do not use outside knowledge.
- Do not invent information.
- Do not explain your reasoning.
- Give a concise answer.
- Mention the relevant page number.
- If the answer is not present, say:
"The information is not available in the provided document."

Question:
{question}

Document Context:
{context}

Give only the final answer.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=False,
        options={
            "num_ctx": 2048,
            "num_predict": 120,
            "temperature": 0.2
        }
    )

    answer = response["message"]["content"].strip()

    # Remove Qwen thinking output if it appears
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()

    if "<think>" in answer:
        answer = answer.replace("<think>", "").strip()

    return answer