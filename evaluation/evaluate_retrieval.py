from statistics import mean

from src.retrieval.comparison_retriever import retrieve_for_comparison

from evaluation.evaluation_utils import (
    expected_paper_coverage,
    hit_at_k,
    paper_matches,
    reciprocal_rank,
    result_payload,
    section_matches,
    serializable_result,
)


def _is_relevant(result, question):
    payload = result_payload(result)
    paper_ok = paper_matches(
        payload.get("paper_title", ""), question.get("expected_paper", "")
    )
    expected_sections = question.get("expected_sections", [])
    section_ok = not expected_sections or any(
        section_matches(payload.get("section", ""), expected)
        for expected in expected_sections
    )
    return paper_ok and section_ok


def _rank_metrics(results, question):
    relevance = [_is_relevant(result, question) for result in results]
    return {
        "hit@1": hit_at_k(relevance, 1),
        "hit@3": hit_at_k(relevance, 3),
        "hit@5": hit_at_k(relevance, 5),
        "mrr": reciprocal_rank(relevance),
    }


def evaluate_single_retrieval(
    question, vector_store, embedder, reranker, retrieve_k=15, rerank_k=5
):
    query_vector = embedder.embed_query(question["question"])
    raw_results = list(vector_store.search(query_vector, limit=retrieve_k))
    reranked = reranker.rerank(
        query=question["question"], results=raw_results, top_k=rerank_k
    )
    before = _rank_metrics(raw_results, question)
    after = _rank_metrics(reranked, question)
    if not before["hit@5"]:
        likely_stage = "RETRIEVAL_FAILURE"
    elif not after["hit@5"]:
        likely_stage = "RERANKING_FAILURE"
    else:
        likely_stage = None
    return {
        "id": question["id"],
        "question": question["question"],
        "type": question["type"],
        "before_reranking": before,
        **after,
        "retrieved_sections": [
            result_payload(result).get("section") for result in reranked
        ],
        "retrieved_papers": [
            result_payload(result).get("paper_title") for result in reranked
        ],
        "raw_results": [serializable_result(item) for item in raw_results[:rerank_k]],
        "reranked_results": [serializable_result(item) for item in reranked],
        "likely_failure_stage": likely_stage,
    }


def evaluate_comparison_retrieval(
    question, vector_store, embedder, reranker, retrieve_k=10, top_k=3
):
    targets = [
        {"query": question["question"], "paper_title": paper}
        for paper in question["expected_papers"]
    ]
    results = retrieve_for_comparison(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        targets=targets,
        per_query_limit=retrieve_k,
        per_query_top_k=top_k,
    )
    papers = [result_payload(item).get("paper_title", "") for item in results]
    coverage = expected_paper_coverage(papers, question["expected_papers"])
    first_ranks = {}
    for expected in question["expected_papers"]:
        first_ranks[expected] = next(
            (
                rank for rank, actual in enumerate(papers, 1)
                if paper_matches(actual, expected)
            ),
            None,
        )
    return {
        "id": question["id"],
        "question": question["question"],
        "type": "comparison",
        "comparison_paper_coverage": coverage,
        "paper_first_ranks": first_ranks,
        "retrieved_papers": papers,
        "retrieved_sections": [
            result_payload(item).get("section") for item in results
        ],
        "reranked_results": [serializable_result(item) for item in results],
        "likely_failure_stage": (
            None if coverage == 1.0 else "PAPER_ROUTING_FAILURE"
        ),
    }


def evaluate_retrieval_dataset(
    questions, vector_store, embedder, reranker, retrieve_k=15, rerank_k=5
):
    results = []
    for question in questions:
        if question["type"] == "comparison":
            results.append(evaluate_comparison_retrieval(
                question, vector_store, embedder, reranker
            ))
        elif question["type"] != "insufficient_evidence":
            results.append(evaluate_single_retrieval(
                question, vector_store, embedder, reranker,
                retrieve_k=retrieve_k, rerank_k=rerank_k,
            ))
    return {"per_question": results, "summary": summarize_retrieval_metrics(results)}


def summarize_retrieval_metrics(results):
    standard = [item for item in results if item["type"] != "comparison"]
    comparisons = [item for item in results if item["type"] == "comparison"]

    def average(items, path, default=0.0):
        values = []
        for item in items:
            value = item
            for key in path:
                value = value[key]
            values.append(value)
        return mean(values) if values else default

    before = {
        metric: average(standard, ("before_reranking", metric))
        for metric in ("hit@1", "hit@3", "hit@5", "mrr")
    }
    after = {
        metric: average(standard, (metric,))
        for metric in ("hit@1", "hit@3", "hit@5", "mrr")
    }
    return {
        "questions": len(standard),
        "comparison_questions": len(comparisons),
        "before_reranking": before,
        "after_reranking": after,
        "reranking_improvement": {
            metric: after[metric] - before[metric]
            for metric in ("hit@1", "hit@3", "mrr")
        },
        "comparison_paper_coverage": average(
            comparisons, ("comparison_paper_coverage",)
        ),
    }
