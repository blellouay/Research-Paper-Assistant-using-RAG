import json
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


def normalize_text(value):
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value):
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "uses", "using", "with",
    }
    aliases = {
        "structure": "architecture", "structured": "architecture",
        "contains": "use", "contain": "use", "have": "use", "has": "use",
        "based": "base", "results": "result", "layers": "layer",
    }
    return {
        aliases.get(token.rstrip("s"), token.rstrip("s"))
        for token in normalize_text(value).split()
        if token not in stop
    }


def semantic_match(first, second, threshold=0.62):
    """Conservative lexical-semantic match for expected fact coverage."""
    first_norm = normalize_text(first)
    second_norm = normalize_text(second)
    if not first_norm or not second_norm:
        return False
    if first_norm in second_norm or second_norm in first_norm:
        return True
    first_tokens = _tokens(first)
    second_tokens = _tokens(second)
    if not first_tokens or not second_tokens:
        return False
    containment = len(first_tokens & second_tokens) / min(
        len(first_tokens), len(second_tokens)
    )
    sequence = SequenceMatcher(None, first_norm, second_norm).ratio()
    return containment >= threshold or sequence >= 0.78


def section_matches(actual, expected):
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)
    if not actual_norm or not expected_norm:
        return False
    return (
        actual_norm == expected_norm
        or expected_norm in actual_norm
        or actual_norm in expected_norm
    )


def paper_matches(actual, expected):
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)
    return actual_norm == expected_norm or (
        min(len(actual_norm), len(expected_norm)) >= 12
        and (actual_norm in expected_norm or expected_norm in actual_norm)
    )


def hit_at_k(relevance, k):
    return int(any(relevance[:k]))


def reciprocal_rank(relevance):
    return next((1.0 / rank for rank, relevant in enumerate(relevance, 1) if relevant), 0.0)


def expected_paper_coverage(retrieved_papers, expected_papers):
    if not expected_papers:
        return 0.0
    matched = sum(
        any(paper_matches(actual, expected) for actual in retrieved_papers)
        for expected in expected_papers
    )
    return matched / len(expected_papers)


def result_payload(result):
    if isinstance(result, dict):
        return result.get("payload", {})
    return getattr(result, "payload", {})


def serializable_result(result):
    payload = dict(result_payload(result))
    if isinstance(result, dict):
        score = result.get("score")
        original_score = result.get("original_score")
    else:
        score = getattr(result, "score", None)
        original_score = None
    return {"score": score, "original_score": original_score, "payload": payload}


def calculate_fact_coverage(expected_facts, final_facts):
    if not expected_facts:
        return None, []
    generated = [fact.get("text", "") for fact in final_facts]
    matches = [
        any(semantic_match(expected, actual) for actual in generated)
        for expected in expected_facts
    ]
    missing = [fact for fact, matched in zip(expected_facts, matches) if not matched]
    return sum(matches) / len(matches), missing


def validate_citations(final_answer, final_facts, available_sources):
    citations = re.findall(r"\[Source\s+(\d+)\]", final_answer, flags=re.I)
    normalized = [f"Source {int(number)}" for number in citations]
    fact_sources = {
        source for fact in final_facts for source in fact.get("sources", [])
    }
    valid = sum(
        source in available_sources and source in fact_sources
        for source in normalized
    )
    missing_all = bool(final_facts) and not normalized
    invalid = [
        source for source in normalized
        if source not in available_sources or source not in fact_sources
    ]
    if missing_all:
        invalid.append("MISSING_CITATIONS")
    return {
        "total_citations": len(normalized),
        "valid_citations": valid,
        "citation_validity": valid / len(normalized) if normalized else (0.0 if missing_all else 1.0),
        "invalid_citations": invalid,
    }


def unsupported_claim_leakage(final_answer, unsupported_claims):
    answer = normalize_text(final_answer)
    leaked = []
    for claim in unsupported_claims:
        text = normalize_text(claim.get("text", ""))
        if text and len(text.split()) >= 4 and text in answer:
            leaked.append(claim.get("text", ""))
    return leaked


def is_abstention(answer):
    normalized = normalize_text(answer)
    phrases = (
        "do not provide enough information",
        "does not provide enough information",
        "insufficient evidence",
        "not enough information",
        "cannot answer from the retrieved",
    )
    return any(phrase in normalized for phrase in phrases)


def load_questions(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON array")
    return questions


def save_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
