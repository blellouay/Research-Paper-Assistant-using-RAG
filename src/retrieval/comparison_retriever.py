def retrieve_for_comparison(
    vector_store,
    embedder,
    reranker,
    targets,
    per_query_limit=8,
    per_query_top_k=3
):
    combined = []

    for target in targets:
        subquery = target["query"]
        paper_title = target["paper_title"]

        query_embedding = embedder.embed_query(
            subquery
        )

        results = vector_store.search_by_paper(
            query_vector=query_embedding,
            paper_title=paper_title,
            limit=per_query_limit
        )

        reranked = reranker.rerank(
            query=subquery,
            results=results,
            top_k=per_query_top_k
        )

        for result in reranked:
            result["retrieval_query"] = subquery
            result["target_paper"] = paper_title

            combined.append(result)

    return combined