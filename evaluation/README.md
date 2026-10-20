# RAG evaluation

This package evaluates retrieval, reranking, grounded generation, verification,
citations, comparison coverage, and abstention without changing the production
pipeline.

## Prerequisites

- Qdrant is running at `http://localhost:6333`.
- The indexed collection exists (default: `research_papers_multi`).
- Ollama is running and `llama3.2:3b` is installed.
- The project virtual environment contains the same dependencies used by the
  notebook.

## Run

From the repository root:

```powershell
python -m evaluation.run_evaluation
```

Optional configuration:

```powershell
python -m evaluation.run_evaluation `
  --collection research_papers_multi `
  --qdrant-url http://localhost:6333 `
  --ollama-model llama3.2:3b
```

The command writes predictable `*_results.json` files and timestamped history
files under `evaluation/results/`.

## Tests

Utility tests do not require Qdrant or Ollama:

```powershell
python -m unittest tests.test_evaluation -v
```

The expected-fact matcher is deliberately conservative and deterministic. It
normalizes morphology and a small set of architectural synonyms, but it is not
an LLM judge. Review missing-fact failures before treating them as definitive.
