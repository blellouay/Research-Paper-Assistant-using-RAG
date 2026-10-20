import re


BERT_PAPER = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
TRANSFORMER_PAPER = "Attention Is All You Need"
PAPER_ENTITY_MAP = {
    "BERT": BERT_PAPER,
    "Transformer": TRANSFORMER_PAPER,
    "original Transformer": TRANSFORMER_PAPER,
    "BERT BASE": BERT_PAPER,
    "BERT LARGE": BERT_PAPER,
}


def canonicalize_claim(text):
    """Create a stable key for duplicate factual claims."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.rstrip(".!?;: ")


def resolve_claim_entity(text):
    """Resolve the primary model named by one already-atomic claim."""
    value = re.sub(r"\s+", " ", str(text)).strip()
    if re.search(r"\bGPT(?:\s+Transformer)?\b", value, flags=re.IGNORECASE):
        return "GPT"
    if re.search(r"\bBERT[ -]?LARGE\b", value, flags=re.IGNORECASE):
        return "BERT LARGE"
    if re.search(r"\bBERT[ -]?BASE\b", value, flags=re.IGNORECASE):
        return "BERT BASE"
    if re.search(r"\bBERT\b", value, flags=re.IGNORECASE):
        return "BERT"
    if re.search(r"\boriginal Transformer\b", value, flags=re.IGNORECASE):
        return "Transformer"
    if re.search(r"\b(?:the\s+)?Transformer\b", value, flags=re.IGNORECASE):
        return "Transformer"
    return None


def resolve_claim_entities(text):
    """Return every explicitly named model while preserving model distinctions."""
    value = re.sub(r"\s+", " ", str(text)).strip()
    entities = []
    if re.search(r"\bGPT(?:\s+Transformer)?\b", value, flags=re.IGNORECASE):
        entities.append("GPT")
    if re.search(r"\bBERT[ -]?LARGE\b", value, flags=re.IGNORECASE):
        entities.append("BERT LARGE")
    elif re.search(r"\bBERT[ -]?BASE\b", value, flags=re.IGNORECASE):
        entities.append("BERT BASE")
    elif re.search(r"\bBERT\b", value, flags=re.IGNORECASE):
        entities.append("BERT")
    transformer_without_named_variant = re.sub(
        r"\b(?:BERT|GPT)(?:[ -]?(?:BASE|LARGE))?\s+Transformer\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    explicit_original = re.search(
        r"\boriginal Transformer\b|\band\s+(?:the\s+)?Transformer\b|"
        r"\b(?:whereas|while)\s+(?:the\s+)?Transformer\b",
        transformer_without_named_variant,
        flags=re.IGNORECASE,
    )
    generic_transformer = re.search(
        r"\b(?:the\s+)?Transformer\b",
        transformer_without_named_variant,
        flags=re.IGNORECASE,
    )
    if explicit_original or (generic_transformer and not any(
        entity.startswith("BERT") or entity == "GPT" for entity in entities
    )):
        entities.append("Transformer")
    return entities


def classify_claim_relation(text):
    """Classify asserted subjects separately from referenced architectures."""
    value = re.sub(r"\s+", " ", str(text)).strip()
    entities = resolve_claim_entities(value)
    if re.search(r"\bwhereas\b|\bwhile\b|\bcompared\s+(?:with|to)\b", value, re.I):
        relation_type = "COMPARATIVE"
    elif re.search(r"\bboth\b", value, re.I) or re.match(
        r"^(?:BERT|Transformer).+\band\s+(?:the\s+)?(?:original\s+)?Transformer\b",
        value,
        re.I,
    ):
        relation_type = "JOINT"
    else:
        relation_type = "SINGLE_SUBJECT"

    primary = resolve_claim_entity(value)
    if re.match(r"^BERT\b", value, re.I):
        primary = next(
            (entity for entity in entities if entity.startswith("BERT")), "BERT"
        )
    elif re.match(r"^(?:The\s+)?(?:original\s+)?Transformer\b", value, re.I):
        primary = "Transformer"
    elif re.match(r"^(?:The\s+)?GPT(?:\s+Transformer)?\b", value, re.I):
        primary = "GPT"

    referenced = [entity for entity in entities if entity != primary]
    if relation_type == "SINGLE_SUBJECT" and referenced:
        relation_type = "REFERENCE_ONLY"
    asserted = entities if relation_type in {"JOINT", "COMPARATIVE"} else ([primary] if primary else [])
    return {
        "primary_entity": primary,
        "referenced_entities": referenced,
        "asserted_entities": asserted,
        "relation_type": relation_type,
    }


def resolve_paper_hint(text):
    """Map an atomic claim's primary entity to its authoritative paper."""
    classification = classify_claim_relation(text)
    if classification["relation_type"] in {"JOINT", "COMPARATIVE"}:
        return None
    entity = classification["primary_entity"]
    if entity == "GPT":
        return None
    return PAPER_ENTITY_MAP.get(entity)


def paper_titles_compatible(actual, expected):
    """Accept exact production titles plus unambiguous legacy short titles."""
    actual_key = canonicalize_claim(actual)
    expected_key = canonicalize_claim(expected)
    if actual_key == expected_key:
        return True
    aliases = {
        canonicalize_claim(BERT_PAPER): {"bert"},
        canonicalize_claim(TRANSFORMER_PAPER): {"transformer"},
    }
    return actual_key in aliases.get(expected_key, set())


def evidence_matches_paper(result, paper_hint):
    if not paper_hint:
        return True
    payload = result.get("payload", {}) if isinstance(result, dict) else result.payload
    return paper_titles_compatible(payload.get("paper_title", ""), paper_hint)


def retrieve_evidence_for_claim(
    claim,
    vector_store,
    embedder,
    reranker,
    paper_hint=None,
    retrieve_k=10,
    evidence_k=3,
):
    """Retrieve and rerank evidence specifically for one atomic claim."""
    query_embedding = embedder.embed_query(claim)

    if paper_hint:
        candidates = vector_store.search_by_paper(
            query_vector=query_embedding,
            paper_title=paper_hint,
            limit=retrieve_k,
        )
    else:
        candidates = vector_store.search(query_embedding, limit=retrieve_k)

    if paper_hint:
        candidates = [
            result for result in candidates
            if evidence_matches_paper(result, paper_hint)
        ]
    if not candidates:
        return []

    technical = re.search(
        r"wordpiece|vocabulary|hidden size|attention heads?|sentence [ab]|"
        r"segment embedding|bert[ -]?(?:base|large)",
        claim,
        flags=re.IGNORECASE,
    )
    reranked = reranker.rerank(
        query=claim,
        results=candidates,
        top_k=retrieve_k if technical else evidence_k,
    )
    if technical:
        preferred_sections = (
            "bert", "input/output representations", "model architecture",
            "pre-training bert",
        )
        reranked.sort(
            key=lambda item: (
                any(
                    name in str(item["payload"].get("section", "")).lower()
                    for name in preferred_sections
                ),
                item.get("score", 0.0),
            ),
            reverse=True,
        )
    return reranked[:evidence_k]
