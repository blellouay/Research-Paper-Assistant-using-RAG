import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class VectorStore:

    def __init__(
        self,
        collection_name="research_papers",
        url="http://localhost:6333"
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    @staticmethod
    def _to_vector(vector):
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def create_collection(self, vector_size):
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def add_chunks(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=self._to_vector(embedding),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document": chunk["document"],
                    "paper_title": chunk["paper_title"],
                    "section": chunk["section"],
                    "text": chunk["text"]
                }
            )

            points.append(point)

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def search(self, query_vector, limit=5):
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=self._to_vector(query_vector),
            limit=limit
        )

        return result.points

    def search_by_paper(
        self,
        query_vector,
        paper_title,
        limit=5
    ):
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=self._to_vector(query_vector),
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="paper_title",
                        match=MatchValue(
                            value=paper_title
                        )
                    )
                ]
            ),
            limit=limit
        )

        return result.points
