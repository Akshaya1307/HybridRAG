import numpy as np


def cosine_similarity(query_embedding, embeddings):
    """
    Calculate cosine similarity between a query
    embedding and all document chunk embeddings.
    """

    query_embedding = np.array(query_embedding)

    embeddings = np.array(embeddings)

    similarities = np.dot(embeddings, query_embedding)

    return similarities


def retrieve_top_k(query, chunks, embeddings, model, k=5):
    """
    Retrieve the top-k most relevant chunks for a query.
    """

    # Convert the query into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Calculate similarity
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )

    # Get indices of highest scores
    top_indices = np.argsort(similarities)[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "chunk": chunks[index],
            "score": float(similarities[index])
        })

    return results