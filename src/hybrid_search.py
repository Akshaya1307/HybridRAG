def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    chunks = {}
    semantic_ranks = {}
    bm25_ranks = {}

    # Store semantic retrieval rankings
    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["chunk"]["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + rank))
        chunks[chunk_id] = result["chunk"]
        semantic_ranks[chunk_id] = rank

    # Store BM25 retrieval rankings
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk"]["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + rank))
        chunks[chunk_id] = result["chunk"]
        bm25_ranks[chunk_id] = rank

    # Rank chunks using combined RRF score
    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    results = []

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        results.append({
            "rank": rank,
            "chunk": chunks[chunk_id],
            "rrf_score": scores[chunk_id],
            "semantic_rank": semantic_ranks.get(chunk_id),
            "bm25_rank": bm25_ranks.get(chunk_id)
        })

    return results