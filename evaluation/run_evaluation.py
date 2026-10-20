import argparse
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

from evaluation.evaluate_generation import evaluate_generation_dataset
from evaluation.evaluate_retrieval import evaluate_retrieval_dataset
from evaluation.evaluation_utils import load_questions, save_json, utc_timestamp


ROOT = Path(__file__).resolve().parent


def _percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def print_summary(retrieval, generation):
    retrieval_summary = retrieval["summary"]
    generation_summary = generation["summary"]
    before = retrieval_summary["before_reranking"]
    after = retrieval_summary["after_reranking"]
    improvement = retrieval_summary["reranking_improvement"]

    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Questions evaluated: {retrieval_summary['questions']}")
    print(f"Before reranking Hit@1: {_percent(before['hit@1'])}")
    print(f"Before reranking Hit@3: {_percent(before['hit@3'])}")
    print(f"Before reranking MRR: {before['mrr']:.3f}")
    print(f"After reranking Hit@1: {_percent(after['hit@1'])}")
    print(f"After reranking Hit@3: {_percent(after['hit@3'])}")
    print(f"After reranking Hit@5: {_percent(after['hit@5'])}")
    print(f"After reranking MRR: {after['mrr']:.3f}")
    print(f"Hit@1 improvement: {improvement['hit@1']:+.3f}")
    print(f"MRR improvement: {improvement['mrr']:+.3f}")
    print(
        "Comparison paper coverage: "
        f"{_percent(retrieval_summary['comparison_paper_coverage'])}"
    )

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION")
    print("=" * 60)
    print(f"Questions evaluated: {generation_summary['questions']}")
    print(f"Fully supported claim rate: {_percent(generation_summary['fully_supported_rate'])}")
    print(f"Partially supported rate: {_percent(generation_summary['partially_supported_rate'])}")
    print(f"Unsupported rate: {_percent(generation_summary['unsupported_rate'])}")
    print(f"Citation validity: {_percent(generation_summary['citation_validity'])}")
    print(
        "Unsupported claim leakage: "
        f"{generation_summary['unsupported_claim_leakage_count']}"
    )
    print(f"Expected fact coverage: {_percent(generation_summary['expected_fact_coverage'])}")
    print(f"Abstention accuracy: {_percent(generation_summary['abstention_accuracy'])}")
    print(
        "Comparison final-paper coverage: "
        f"{_percent(generation_summary['comparison_final_paper_coverage'])}"
    )

    failures = []
    for item in retrieval["per_question"] + generation["per_question"]:
        if item.get("likely_failure_stage"):
            failures.append(item)
    print("\n" + "=" * 60)
    print("FAILURES")
    print("=" * 60)
    if not failures:
        print("No failures detected by the deterministic checks.")
    for item in failures:
        print(f"\n{item['id']}")
        print(f"Question: {item['question']}")
        print(f"Likely stage: {item['likely_failure_stage']}")
        if item.get("missing_expected_facts"):
            print("Missing expected facts:")
            for fact in item["missing_expected_facts"]:
                print(f"- {fact}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the research-paper RAG pipeline")
    parser.add_argument("--questions", default=str(ROOT / "questions.json"))
    parser.add_argument("--collection", default="research_papers_multi")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    embedder = Embedder()
    vector_store = VectorStore(
        collection_name=args.collection, url=args.qdrant_url
    )
    reranker = Reranker()

    retrieval = evaluate_retrieval_dataset(
        questions, vector_store, embedder, reranker
    )
    generation = evaluate_generation_dataset(
        questions, vector_store, embedder, reranker, model=args.ollama_model
    )
    summary = {
        "retrieval": retrieval["summary"],
        "generation": generation["summary"],
    }

    results_dir = Path(args.results_dir)
    timestamp = utc_timestamp()
    for name, value in (
        ("retrieval_results", retrieval),
        ("generation_results", generation),
        ("summary", summary),
    ):
        save_json(results_dir / f"{name}.json", value)
        save_json(results_dir / f"{name}_{timestamp}.json", value)
    print_summary(retrieval, generation)
    print(f"\nResults saved to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
