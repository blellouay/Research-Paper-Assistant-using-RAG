import contextlib
import io
from statistics import mean

from src.generation.generator import generate_grounded_answer
from src.retrieval.comparison_retriever import retrieve_for_comparison

from evaluation.evaluation_utils import (
    calculate_fact_coverage,
    expected_paper_coverage,
    is_abstention,
    paper_matches,
    semantic_match,
    serializable_result,
    unsupported_claim_leakage,
    validate_citations,
)


def _retrieve_for_generation(
    question, vector_store, embedder, reranker, retrieve_k=15, top_k=5
):
    if question["type"] == "comparison":
        targets = [
            {"query": question["question"], "paper_title": paper}
            for paper in question["expected_papers"]
        ]
        return retrieve_for_comparison(
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            targets=targets,
            per_query_limit=retrieve_k,
            per_query_top_k=3,
        )
    query_vector = embedder.embed_query(question["question"])
    candidates = vector_store.search(query_vector, limit=retrieve_k)
    return reranker.rerank(
        query=question["question"], results=candidates, top_k=top_k
    )


def calculate_claim_metrics(result):
    verified = result.get("verified_claims", [])
    total = len(verified)
    fully = len(result.get("supported_claims", []))
    partial = len(result.get("partially_supported_claims", []))
    unsupported = len(result.get("rejected_claims", []))
    errors = len(result.get("verification_errors", []))
    denominator = total or 1
    return {
        "verified_claims": total,
        "fully_supported_claims": fully,
        "partially_supported_claims": partial,
        "unsupported_claims": unsupported,
        "verification_errors": errors,
        "fully_supported_rate": fully / denominator,
        "partially_supported_rate": partial / denominator,
        "unsupported_rate": unsupported / denominator,
        "verification_error_rate": errors / denominator,
    }


def _source_metadata(result):
    items = []
    for evidence in result.get("evidence_used", {}).values():
        items.extend(evidence)
    return {item["source"]: item for item in items}


def _comparison_metrics(question, result):
    if question["type"] != "comparison":
        return None
    source_metadata = _source_metadata(result)
    fact_sources = {
        source
        for fact in result.get("final_fact_set", [])
        for source in fact.get("sources", [])
    }
    final_papers = [
        metadata["paper"]
        for source, metadata in source_metadata.items()
        if source in fact_sources
    ]
    coverage = expected_paper_coverage(final_papers, question["expected_papers"])
    per_paper = {
        expected: any(paper_matches(actual, expected) for actual in final_papers)
        for expected in question["expected_papers"]
    }
    return {
        "final_answer_paper_coverage": coverage,
        "supported_facts_by_expected_paper": per_paper,
        "has_all_expected_papers": coverage == 1.0,
    }


def _likely_failure_stage(question, result, citation, leakage, missing_facts, comparison):
    if question.get("expected_answerable") is False and not is_abstention(
        result["final_answer"]
    ):
        return "ABSTENTION_FAILURE"
    if citation["invalid_citations"]:
        return "CITATION_FAILURE"
    if leakage:
        return "SYNTHESIS_FAILURE"
    if comparison and not comparison["has_all_expected_papers"]:
        evidence_papers = [
            item["paper"] for item in _source_metadata(result).values()
        ]
        retrieval_coverage = expected_paper_coverage(
            evidence_papers, question["expected_papers"]
        )
        return (
            "PAPER_ROUTING_FAILURE"
            if retrieval_coverage < 1.0
            else "VERIFICATION_FALSE_NEGATIVE"
        )
    if missing_facts:
        rejected = result.get("rejected_claims", [])
        if any(
            semantic_match(expected, claim.get("text", ""))
            for expected in missing_facts for claim in rejected
        ):
            return "VERIFICATION_FALSE_NEGATIVE"
        return "GENERATION_FAILURE"
    if any(
        not claim.get("sources") for claim in result.get("supported_claims", [])
    ):
        return "VERIFICATION_FALSE_POSITIVE"
    return None


def evaluate_single_generation(
    question,
    vector_store,
    embedder,
    reranker,
    model="llama3.2:3b",
    retrieve_k=15,
    top_k=5,
    show_pipeline_debug=False,
):
    retrieval_results = _retrieve_for_generation(
        question, vector_store, embedder, reranker,
        retrieve_k=retrieve_k, top_k=top_k,
    )
    output_buffer = io.StringIO()
    stream = contextlib.nullcontext() if show_pipeline_debug else contextlib.redirect_stdout(output_buffer)
    with stream:
        result = generate_grounded_answer(
            question=question["question"],
            reranked_results=retrieval_results,
            model=model,
            top_k=len(retrieval_results),
            comparison=question["type"] == "comparison",
            debug=True,
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
        )

    claim_metrics = calculate_claim_metrics(result)
    available_sources = set(_source_metadata(result))
    citation = validate_citations(
        result["final_answer"], result.get("final_fact_set", []), available_sources
    )
    leakage = unsupported_claim_leakage(
        result["final_answer"], result.get("rejected_claims", [])
    )
    fact_coverage, missing_facts = calculate_fact_coverage(
        question.get("expected_facts", []), result.get("final_fact_set", [])
    )
    comparison = _comparison_metrics(question, result)
    abstained = is_abstention(result["final_answer"])
    expected_abstention = question.get("expected_answerable") is False
    abstention_correct = abstained if expected_abstention else None
    likely_stage = _likely_failure_stage(
        question, result, citation, leakage, missing_facts, comparison
    )

    return {
        "id": question["id"],
        "question": question["question"],
        "type": question["type"],
        "draft_answer": result.get("draft_answer"),
        "final_answer": result["final_answer"],
        "retrieved_sources": [
            serializable_result(item) for item in result.get("sources", [])
        ],
        "extracted_claims": result.get("extracted_claims", []),
        "supported_claims": result.get("supported_claims", []),
        "partially_supported_claims": result.get("partially_supported_claims", []),
        "unsupported_claims": result.get("rejected_claims", []),
        "verification_errors": result.get("verification_errors", []),
        "final_fact_set": result.get("final_fact_set", []),
        "claim_metrics": claim_metrics,
        **citation,
        "unsupported_claim_leakage_count": len(leakage),
        "leaked_unsupported_claims": leakage,
        "expected_fact_coverage": fact_coverage,
        "missing_expected_facts": missing_facts,
        "abstained": abstained,
        "abstention_correct": abstention_correct,
        "comparison_metrics": comparison,
        "likely_failure_stage": likely_stage,
    }


def evaluate_generation_dataset(
    questions, vector_store, embedder, reranker, model="llama3.2:3b"
):
    results = [
        evaluate_single_generation(
            question, vector_store, embedder, reranker, model=model
        )
        for question in questions
    ]
    return {"per_question": results, "summary": summarize_generation_metrics(results)}


def summarize_generation_metrics(results):
    claim_totals = {
        key: sum(item["claim_metrics"][key] for item in results)
        for key in (
            "verified_claims", "fully_supported_claims",
            "partially_supported_claims", "unsupported_claims",
            "verification_errors",
        )
    }
    verified = claim_totals["verified_claims"] or 1
    coverage_values = [
        item["expected_fact_coverage"] for item in results
        if item["expected_fact_coverage"] is not None
    ]
    abstentions = [
        item["abstention_correct"] for item in results
        if item["abstention_correct"] is not None
    ]
    comparisons = [
        item["comparison_metrics"]["final_answer_paper_coverage"]
        for item in results if item["comparison_metrics"] is not None
    ]
    return {
        "questions": len(results),
        **claim_totals,
        "fully_supported_rate": claim_totals["fully_supported_claims"] / verified,
        "partially_supported_rate": claim_totals["partially_supported_claims"] / verified,
        "unsupported_rate": claim_totals["unsupported_claims"] / verified,
        "citation_validity": mean(
            item["citation_validity"] for item in results
        ) if results else 1.0,
        "unsupported_claim_leakage_count": sum(
            item["unsupported_claim_leakage_count"] for item in results
        ),
        "expected_fact_coverage": mean(coverage_values) if coverage_values else None,
        "abstention_accuracy": mean(abstentions) if abstentions else None,
        "comparison_final_paper_coverage": mean(comparisons) if comparisons else None,
    }
