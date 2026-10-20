from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query):
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding