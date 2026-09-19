from rank_bm25 import BM25Okapi


def build_bm25_index(chunks):
    """
    Build a BM25 index from document chunks.
    """

    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def bm25_search(query, chunks, bm25, k=5):
    """
    Retrieve the top-k chunks using BM25 keyword search.
    """

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "chunk": chunks[index],
            "score": float(scores[index])
        })

    return results