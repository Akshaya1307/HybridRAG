from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(chunks, model):
    """
    Create embeddings for all chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings