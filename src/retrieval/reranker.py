from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k=5):
        pairs = []

        for result in results:
            text = result.payload["text"]
            pairs.append([query, text])

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            reranked.append({
                "score": float(score),
                "original_score": result.score,
                "payload": result.payload
            })

        reranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked[:top_k]