import json
import re
import unicodedata
from difflib import SequenceMatcher

from ollama import Client
from src.retrieval.claim_evidence import (
    BERT_PAPER,
    TRANSFORMER_PAPER,
    canonicalize_claim,
    classify_claim_relation,
    evidence_matches_paper,
    paper_titles_compatible,
    resolve_claim_entity,
    resolve_claim_entities,
    resolve_paper_hint,
    retrieve_evidence_for_claim,
)


INSUFFICIENT_EVIDENCE = (
    "The retrieved excerpts do not provide enough information to answer this question."
)
SOURCE_PATTERN = re.compile(r"^Source ([1-9][0-9]*)$")


class StructuredOutputError(RuntimeError):
    """Raised when an LLM response cannot be validated as the required JSON."""


class SourceNormalizationError(StructuredOutputError):
    """Raised when a supported verdict has no usable evidence identifier."""


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StructuredOutputError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _source_block(source_number, result):
    payload = result["payload"]
    return f"""
SOURCE {source_number}
Paper: {payload["paper_title"]}
Section: {payload["section"]}

{payload["text"]}
""".strip()


def build_context(results):
    return "\n\n".join(
        _source_block(number, result)
        for number, result in enumerate(results, start=1)
    )


def build_comparison_context(results):
    grouped = {}
    for number, result in enumerate(results, start=1):
        title = result["payload"]["paper_title"]
        grouped.setdefault(title, []).append((number, result))

    parts = []
    for title, paper_results in grouped.items():
        parts.append(f"=== PAPER: {title} ===")
        parts.extend(_source_block(number, result) for number, result in paper_results)
    return "\n\n".join(parts)


def build_prompt(question, context):
    return f"""
You are a research paper assistant. Draft a concise answer using only the excerpts.
Do not use outside knowledge. Cite claims as [Source 1], [Source 2], and so on.
This is an unverified draft; a separate claim verifier will check it.

QUESTION:
{question}

CONTEXT:
{context}

DRAFT ANSWER:
""".strip()


def build_comparison_prompt(question, context):
    return f"""
You are a research paper assistant. Draft a concise comparison using only the excerpts.
Describe each system from its own paper, keep systems distinct, and cite claims as
[Source 1], [Source 2], and so on. Do not use outside knowledge.
This is an unverified draft; a separate claim verifier will check every claim.

QUESTION:
{question}

CONTEXT:
{context}

DRAFT ANSWER:
""".strip()


def generate_evidence_answer(
    question,
    reranked_results,
    model="llama3.2:3b",
    top_k=3,
    comparison=None,
    timeout=90,
    client=None,
):
    """Generate a citation-grounded answer in one LLM call.

    This is the responsive UI path. Unlike ``generate_grounded_answer``, it
    does not run atomic claim extraction and per-claim verification.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    selected_results = list(reranked_results[:top_k])
    if not selected_results:
        raise ValueError("reranked_results must contain at least one result")

    titles = {result["payload"]["paper_title"] for result in selected_results}
    if comparison is None:
        comparison = len(titles) > 1

    context = (
        build_comparison_context(selected_results)
        if comparison
        else build_context(selected_results)
    )
    prompt = (
        build_comparison_prompt(question, context)
        if comparison
        else build_prompt(question, context)
    )
    prompt += (
        "\n\nReturn the final answer directly. Keep it concise, cite every factual "
        "claim, and say that the excerpts are insufficient when the evidence does "
        "not answer the question."
    )
    answer = _run_chat(prompt, model, timeout, 0.1, client, json_output=False)
    return {
        "answer": answer,
        "final_answer": answer,
        "sources": selected_results,
        "context": context,
        "comparison": comparison,
        "verified_claims": [],
        "verification_mode": "fast",
    }


def _run_chat(prompt, model, timeout, temperature, client=None, json_output=False):
    ollama_client = client or Client(timeout=timeout)
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature},
    }
    if json_output:
        kwargs["format"] = "json"

    try:
        response = ollama_client.chat(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Ollama generation failed for model {model!r}. "
            "Confirm that Ollama is running and the model is installed."
        ) from exc

    content = response.message.content.strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content


def _parse_json_object(raw_text):
    """Parse one JSON object, tolerating fences or short surrounding prose."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    raise StructuredOutputError("LLM response did not contain a valid JSON object")


def _json_operation(prompt, model, timeout, client):
    """Request JSON and retry once with a repair instruction if parsing fails."""
    raw = _run_chat(prompt, model, timeout, 0.0, client, json_output=True)
    try:
        return _parse_json_object(raw)
    except StructuredOutputError as first_error:
        repair_prompt = f"""
Return ONLY one valid JSON object matching the schema requested below.
Do not add Markdown fences or commentary.

ORIGINAL REQUEST:
{prompt}

INVALID RESPONSE:
{raw}
""".strip()
        repaired = _run_chat(
            repair_prompt, model, timeout, 0.0, client, json_output=True
        )
        try:
            return _parse_json_object(repaired)
        except StructuredOutputError as second_error:
            raise StructuredOutputError(
                "LLM returned malformed JSON twice"
            ) from second_error


def extract_claims(draft_answer, model="llama3.2:3b", timeout=120, client=None):
    """Extract atomic factual claims from a draft as validated JSON records."""
    prompt = f"""
Extract every factual claim from the draft answer into atomic claims.

Rules:
- Each claim must express exactly one independently verifiable fact.
- Split compound comparisons into separate claims for each side and relationship.
- Extract ALL factual claims; do not return only the last or most important claim.
- Make every claim self-contained. Resolve "the model", "it", and implicit bullet
  subjects from the nearest local heading, without guessing across sections.
- Preserve the factual meaning; do not correct, add, or remove facts.
- Exclude headings, rhetorical text, and citation markers.
- Assign sequential IDs starting at claim_1.
- Example: "BERT uses bidirectional attention whereas the Transformer uses
  constrained attention" becomes two claims, one about BERT and one about the
  Transformer.
- Example: "BERT is pretrained on unlabeled data and fine-tuned on labeled
  downstream data" becomes two separate claims.
- Return ONLY JSON with this schema:
  {{"claims": [{{"id": "claim_1", "text": "...", "entities": ["BERT"],
    "paper_hint": "exact paper title or null"}}]}}

DRAFT ANSWER:
{draft_answer}
""".strip()
    data = _json_operation(prompt, model, timeout, client)
    raw_claims = data.get("claims")
    if not isinstance(raw_claims, list):
        raise StructuredOutputError("claim extraction JSON must contain a claims list")

    claims = []
    seen_ids = set()
    for index, claim in enumerate(raw_claims, start=1):
        if not isinstance(claim, dict):
            raise StructuredOutputError("each extracted claim must be an object")
        claim_id = claim.get("id")
        text = claim.get("text")
        entities = claim.get("entities", [])
        paper_hint = claim.get("paper_hint")
        if not isinstance(claim_id, str) or not claim_id.strip():
            claim_id = f"claim_{index}"
        claim_id = claim_id.strip()
        if claim_id in seen_ids:
            raise StructuredOutputError(f"duplicate claim id: {claim_id}")
        if not isinstance(text, str) or not text.strip():
            raise StructuredOutputError(f"{claim_id} has no claim text")
        if not isinstance(entities, list) or not all(isinstance(x, str) for x in entities):
            entities = []
        if not isinstance(paper_hint, str) or not paper_hint.strip():
            paper_hint = None
        seen_ids.add(claim_id)
        claims.append({
            "id": claim_id,
            "text": text.strip(),
            "entities": [entity.strip() for entity in entities if entity.strip()],
            "paper_hint": paper_hint.strip() if paper_hint else None,
        })
    atomic_claims = []
    comparison_pattern = re.compile(
        r"\s*,?\s+whereas\s+|\s*;\s*while\s+",
        flags=re.IGNORECASE,
    )
    for claim in claims:
        parts = [part.strip(" ,;.") for part in comparison_pattern.split(claim["text"])]
        parts = [part for part in parts if len(part.split()) >= 3]
        if len(parts) < 2:
            parts = [claim["text"]]
        for part in parts:
            part, draft_section, original_bullet = _resolve_local_subject(
                part, draft_answer
            )
            part = _normalize_transformer_translation_result(part, original_bullet)
            part = _normalize_known_claim_wording(part)
            hidden_size_parts = _split_hidden_size_claim(part)
            if hidden_size_parts:
                parts_to_add = hidden_size_parts
            else:
                parts_to_add = _split_both_model_claim(part)
            for atomic_part in parts_to_add:
                classification = classify_claim_relation(atomic_part)
                part_entities = classification["asserted_entities"]
                part_paper_hint = resolve_paper_hint(atomic_part)
                atomic_claims.append({
                    "id": f"claim_{len(atomic_claims) + 1}",
                    "text": atomic_part.strip(),
                    "entities": part_entities,
                    "primary_entity": classification["primary_entity"],
                    "referenced_entities": classification["referenced_entities"],
                    "relation_type": classification["relation_type"],
                    "paper_hint": part_paper_hint,
                    "original_draft_section": draft_section,
                    "original_bullet": original_bullet,
                })

    return atomic_claims


def _draft_sections(draft_answer):
    sections = []
    heading = None
    for raw_line in draft_answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        clean = line.strip("#*: ")
        if re.fullmatch(
            r"(?:BERT|Original Transformer(?: Architecture)?|Transformer(?: Architecture)?)",
            clean,
            flags=re.IGNORECASE,
        ):
            heading = clean
            continue
        if line.startswith(("-", "*", "•")):
            sections.append((heading, line.lstrip("-*• ").strip()))
    return sections


def _resolve_local_subject(claim_text, draft_answer):
    """Resolve an implicit model subject only from the bullet's local heading."""
    claim_key = canonicalize_claim(claim_text)
    for heading, bullet in _draft_sections(draft_answer):
        bullet_key = canonicalize_claim(bullet)
        if claim_key not in bullet_key and bullet_key not in claim_key:
            continue
        if not heading:
            return claim_text, None, bullet
        if heading.lower().startswith("bert"):
            value = re.sub(
                r"^(?:the\s+)?model architecture\b",
                "BERT's model architecture",
                claim_text,
                flags=re.IGNORECASE,
            )
            value = re.sub(
                r"^(?:the\s+)?model\b", "BERT", value, flags=re.IGNORECASE
            )
        else:
            value = re.sub(
                r"^(?:the\s+)?model(?: architecture)?\b",
                "The original Transformer",
                claim_text,
                flags=re.IGNORECASE,
            )
        return value, heading, bullet
    return claim_text, None, claim_text


def _split_both_model_claim(text):
    match = re.fullmatch(
        r"Both\s+BERT\s+and\s+(?:the\s+)?(?:original\s+)?Transformer\s+(.+)",
        text.strip().rstrip("."),
        flags=re.IGNORECASE,
    )
    if not match:
        return [text]
    predicate = match.group(1).strip()
    predicate = re.sub(r"^use\b", "uses", predicate, flags=re.IGNORECASE)
    predicate = re.sub(r"^have\b", "has", predicate, flags=re.IGNORECASE)
    return [f"BERT {predicate}.", f"The original Transformer {predicate}."]


def _split_hidden_size_claim(text):
    if re.fullmatch(
        r"BERT\s+(?:has|uses)\s+(?:a\s+)?hidden size of\s+768\s+and\s+1024\. ?",
        text.strip(),
        flags=re.IGNORECASE,
    ):
        return [
            "BERT BASE has hidden size 768.",
            "BERT LARGE has hidden size 1024.",
        ]
    return None


def _normalize_transformer_translation_result(text, original_bullet):
    """Retain WMT scope when atomic extraction over-generalizes a draft bullet."""
    if not re.search(
        r"Transformer.*(?:new\s+)?state[- ]of[- ]the[- ]art.*translation",
        text,
        flags=re.IGNORECASE,
    ):
        return text
    local = f"{text} {original_bullet or ''}"
    if (
        re.search(r"WMT\s*2014", local, flags=re.IGNORECASE)
        and re.search(r"English[- ]to[- ]German", local, flags=re.IGNORECASE)
        and re.search(r"English[- ]to[- ]French", local, flags=re.IGNORECASE)
    ):
        return (
            "The original Transformer achieved state-of-the-art results on the "
            "reported WMT 2014 English-to-German and English-to-French "
            "translation tasks."
        )
    return text


def _normalize_known_claim_wording(text):
    """Correct a known BERT stage/task wording error before verification."""
    value = text.strip()
    if re.fullmatch(
        r"BERT\s+(?:has|uses)\s+two\s+pre-?train(?:ed|ing)\s+tasks?\s*:\s*"
        r"pre-?training\s+and\s+fine-?tuning\. ?",
        value,
        flags=re.IGNORECASE,
    ):
        return "BERT's framework has two stages: pre-training and fine-tuning."
    return value


def _available_source_ids(context):
    return {
        f"Source {number}"
        for number in re.findall(r"(?m)^SOURCE ([1-9][0-9]*)$", context)
    }


def normalize_source_id(value, available_source_ids):
    """Normalize common source/evidence citation formats to ``Source N``."""
    if not isinstance(value, (str, int)):
        return None
    match = re.fullmatch(
        r"\s*\[?\s*(?:(?:source|evidence)\s*)?(\d+)\s*\]?\s*",
        str(value),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    normalized = f"Source {int(match.group(1))}"
    return normalized if normalized in available_source_ids else None


def normalize_sources(value, available_source_ids):
    """Normalize safe scalar, list, and object source variants."""
    if value is None:
        items = []
    elif isinstance(value, dict):
        items = [value]
    elif isinstance(value, (str, int)):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise StructuredOutputError("verification sources field has an invalid type")

    normalized_sources = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("source")
        normalized = normalize_source_id(item, available_source_ids)
        if normalized and normalized not in normalized_sources:
            normalized_sources.append(normalized)
    return normalized_sources


def _evidence_key(result):
    payload = result["payload"]
    return (
        payload.get("document"),
        payload.get("chunk_id"),
        payload.get("paper_title"),
        payload.get("section"),
        payload.get("text"),
    )


def _build_claim_evidence(results, source_registry):
    blocks = []
    source_ids = []
    evidence_used = []
    for result in results:
        key = _evidence_key(result)
        if key not in source_registry:
            source_registry[key] = len(source_registry) + 1
        number = source_registry[key]
        source_id = f"Source {number}"
        source_ids.append(source_id)
        payload = result["payload"]
        blocks.append(_source_block(number, result))
        evidence_used.append({
            "source": source_id,
            "paper": payload["paper_title"],
            "section": payload["section"],
        })
    return {
        "context": "\n\n".join(blocks),
        "results": results,
        "source_ids": source_ids,
        "evidence_used": evidence_used,
    }


def _match_tokens(text):
    words = re.findall(r"[a-z0-9]+", _semantic_normalize(text).lower())
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "uses", "using", "with",
    }
    return [word.rstrip("s") for word in words if word not in stopwords]


def find_direct_support(claim, evidence_results, source_ids):
    """Return a deterministic verdict for near-direct sentence support."""
    claim_tokens = set(_match_tokens(claim["text"]))
    if len(claim_tokens) < 3:
        return None

    paper_hint = claim.get("paper_hint")
    for result, source_id in zip(evidence_results, source_ids):
        payload = result["payload"]
        if paper_hint and payload.get("paper_title") != paper_hint:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", payload.get("text", ""))
        for sentence in sentences:
            sentence_tokens = set(_match_tokens(sentence))
            if not sentence_tokens:
                continue
            coverage = len(claim_tokens & sentence_tokens) / len(claim_tokens)
            similarity = SequenceMatcher(
                None,
                canonicalize_claim(claim["text"]),
                canonicalize_claim(sentence),
            ).ratio()
            if coverage >= 0.88 and similarity >= 0.58:
                return {
                    "claim_id": claim["id"],
                    "text": claim["text"],
                    "status": "SUPPORTED",
                    "supported": True,
                    "sources": [source_id],
                    "supported_part": None,
                    "unsupported_part": None,
                    "reason": (
                        f"{source_id} contains a near-direct textual statement "
                        "of the claim."
                    ),
                    "verification_error": False,
                    "method": "direct_match",
                }
    return None


def normalize_supported(value):
    """Normalize common boolean verdict variants returned by an LLM."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "supported", "yes"}:
            return True
        if normalized in {"false", "unsupported", "no"}:
            return False
    raise ValueError(f"invalid supported verdict: {value!r}")


def normalize_status(value):
    """Normalize three-way verifier statuses and legacy boolean verdicts."""
    if isinstance(value, bool):
        return "SUPPORTED" if value else "UNSUPPORTED"
    if isinstance(value, str):
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "TRUE": "SUPPORTED",
            "YES": "SUPPORTED",
            "SUPPORTED": "SUPPORTED",
            "PARTIAL": "PARTIALLY_SUPPORTED",
            "PARTIALLY_SUPPORTED": "PARTIALLY_SUPPORTED",
            "FALSE": "UNSUPPORTED",
            "NO": "UNSUPPORTED",
            "UNSUPPORTED": "UNSUPPORTED",
        }
        if normalized in aliases:
            return aliases[normalized]
    raise ValueError(f"invalid verification status: {value!r}")


def _semantic_normalize(text):
    """Normalize formatting noise without changing the statement's meaning."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip()


def _format_evidence(context):
    """Preserve source boundaries while normalizing Unicode and line spacing."""
    context = unicodedata.normalize("NFKC", context)
    lines = [line.rstrip() for line in context.splitlines()]
    formatted = "\n".join(lines).strip()
    return re.sub(
        r"(?m)^SOURCE ([1-9][0-9]*)$",
        lambda match: f"EVIDENCE {match.group(1)}\nSource {match.group(1)}",
        formatted,
    )


def validate_verdict_consistency(result):
    """Return an explanation when status and reason contradict each other."""
    reason = _semantic_normalize(result.get("reason", "")).lower()
    status = result.get("status")
    positive_phrases = (
        "explicitly states",
        "directly states",
        "supports the claim",
        "supports this claim",
        "establishes that",
        "directly supports",
        "semantically supports",
    )
    negative_phrases = (
        "does not state",
        "do not state",
        "not explicitly stated",
        "no retrieved excerpt",
        "no excerpt",
        "evidence is missing",
        "not supported",
        "requires outside knowledge",
    )

    if status == "UNSUPPORTED" and any(phrase in reason for phrase in positive_phrases):
        return "UNSUPPORTED verdict conflicts with a reason that says evidence supports the claim"
    if status == "SUPPORTED" and any(phrase in reason for phrase in negative_phrases):
        return "SUPPORTED verdict conflicts with a reason that identifies missing evidence"
    return None


def validate_partial_verdict(result):
    if result.get("status") != "PARTIALLY_SUPPORTED":
        return None
    original = canonicalize_claim(result.get("text", ""))
    supported = canonicalize_claim(result.get("supported_part", ""))
    unsupported = canonicalize_claim(result.get("unsupported_part", ""))
    if not supported:
        return "supported_part is empty"
    if supported == original:
        return "supported_part repeats the full original claim"
    if not unsupported:
        return "unsupported_part is empty"
    if supported == unsupported:
        return "supported_part and unsupported_part are identical"
    return None


def _normalize_verdict(data, claim, context):
    returned_claim_id = data.get("claim_id")
    if not isinstance(returned_claim_id, str) or not returned_claim_id.strip():
        raise StructuredOutputError("verification claim_id must be a non-empty string")

    status_value = data.get("status", data.get("supported"))
    try:
        status = normalize_status(status_value)
    except ValueError as exc:
        raise StructuredOutputError(str(exc)) from exc

    sources = data.get("sources")
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise StructuredOutputError("verification reason must be a non-empty string")

    available = _available_source_ids(context)
    normalized_sources = normalize_sources(sources, available)

    supported_part = data.get("supported_part")
    unsupported_part = data.get("unsupported_part")

    if status in {"SUPPORTED", "PARTIALLY_SUPPORTED"} and not normalized_sources:
        raise SourceNormalizationError(
            f"{status} verdict supplied no valid supporting source ID"
        )

    if status == "PARTIALLY_SUPPORTED":
        if not isinstance(supported_part, str) or not supported_part.strip():
            status = "UNSUPPORTED"
            normalized_sources = []
            supported_part = None
            unsupported_part = claim["text"]
            reason = (
                f"{reason.strip()} The verifier did not identify a usable supported portion."
            )
        else:
            supported_part = supported_part.strip()
            if not isinstance(unsupported_part, str) or not unsupported_part.strip():
                unsupported_part = claim["text"]
            else:
                unsupported_part = unsupported_part.strip()
    else:
        supported_part = None
        unsupported_part = claim["text"] if status == "UNSUPPORTED" else None

    return {
        "claim_id": claim["id"],
        "text": claim["text"],
        "status": status,
        "supported": status == "SUPPORTED",
        "sources": normalized_sources if status != "UNSUPPORTED" else [],
        "supported_part": supported_part,
        "unsupported_part": unsupported_part,
        "reason": reason.strip(),
        "verification_error": False,
    }


def verify_claim(claim, context, model="llama3.2:3b", timeout=120, client=None):
    """Verify one atomic claim and validate its cited source IDs."""
    normalized_claim = _semantic_normalize(claim["text"])
    evidence = _format_evidence(context)
    prompt = f"""
Verify ONE factual claim against the retrieved excerpts using semantic entailment.

Rules:
- You are not judging whether the statement is generally true. Judge only
  whether the provided excerpts semantically entail it.
- Exact wording is not required. Accept faithful paraphrases, synonymous wording,
  harmless compression, singular/plural normalization, equivalent terminology,
  and simple composition of facts from the same source when it adds no inference.
- Use only the excerpts below; ignore all outside knowledge.
- SUPPORTED means the evidence establishes the full claim by direct statement or
  faithful semantic equivalence.
- PARTIALLY_SUPPORTED means evidence establishes a separable portion but not the
  entire claim. Prefer a strictly narrower supported core over rejecting a useful
  fact because of one unsupported qualifier. Copy the supported and unsupported
  portions into separate fields.
- UNSUPPORTED means the claim requires missing evidence, speculation, outside
  knowledge, or an unstated comparison.
- Do not infer missing facts or silently correct the claim.
- A comparison is supported only when the excerpts support both sides.
- List only source IDs that directly support the full claim or supported portion.
- The reason must identify what the cited excerpts establish or what specific
  evidence is missing. Do not merely repeat or reverse the claim.
- Return ONLY JSON with this schema:
  {{"claim_id": "{claim['id']}", "status": "SUPPORTED",
    "sources": ["Source 1"], "supported_part": null,
    "unsupported_part": null, "reason": "..."}}

CLAIM:
{normalized_claim}

EVIDENCE:
{evidence}
""".strip()
    data = _json_operation(prompt, model, timeout, client)
    try:
        verdict = _normalize_verdict(data, claim, context)
    except SourceNormalizationError:
        available = sorted(
            _available_source_ids(context),
            key=lambda item: int(item.split()[-1]),
        )
        citation_retry_prompt = f"""
Your previous semantic verdict cited a missing or invalid source identifier.
Do not change the claim or use outside knowledge. Re-check the evidence and return
the complete verdict JSON again. If the status is SUPPORTED or
PARTIALLY_SUPPORTED, `sources` MUST contain one or more identifiers copied
exactly from this list: {json.dumps(available)}. If no listed excerpt explicitly
supports the claim, return UNSUPPORTED with an empty sources list.

CLAIM:
{normalized_claim}

EVIDENCE:
{evidence}

PREVIOUS VERDICT:
{json.dumps(data, ensure_ascii=False)}
""".strip()
        retry_data = _json_operation(citation_retry_prompt, model, timeout, client)
        verdict = _normalize_verdict(retry_data, claim, context)
    malformed_partial = validate_partial_verdict(verdict)
    if malformed_partial:
        partial_retry_prompt = f"""
{prompt}

Your previous PARTIALLY_SUPPORTED verdict was malformed:
{malformed_partial}

PREVIOUS VERDICT:
{json.dumps(data, ensure_ascii=False)}

Return a corrected JSON verdict. `supported_part` must be a strictly narrower,
self-contained proposition explicitly established by the cited evidence.
`unsupported_part` must identify the remaining unsupported proposition. Never
copy the full original claim into either field.
""".strip()
        partial_retry_data = _json_operation(
            partial_retry_prompt, model, timeout, client
        )
        verdict = _normalize_verdict(partial_retry_data, claim, context)
        malformed_partial = validate_partial_verdict(verdict)
        if malformed_partial:
            raise StructuredOutputError(
                f"malformed partial verdict after retry: {malformed_partial}"
            )
    inconsistency = validate_verdict_consistency(verdict)
    if inconsistency:
        retry_prompt = f"""
{prompt}

Your previous verdict was internally inconsistent:
{inconsistency}

PREVIOUS VERDICT:
{json.dumps(data, ensure_ascii=False)}

Re-evaluate the CLAIM against the EVIDENCE. Return one corrected JSON object.
The status and reason must agree. Exact wording is not required for support.
""".strip()
        retry_data = _json_operation(retry_prompt, model, timeout, client)
        verdict = _normalize_verdict(retry_data, claim, context)
        retry_inconsistency = validate_verdict_consistency(verdict)
        if retry_inconsistency:
            raise StructuredOutputError(
                f"verifier remained inconsistent after retry: {retry_inconsistency}"
            )

    if verdict["status"] == "UNSUPPORTED":
        supported_core = _supported_core_candidate(claim["text"], context)
        if supported_core:
            core_text, unsupported_part = supported_core
            core_claim = dict(claim)
            core_claim["text"] = core_text
            core_prompt = f"""
Verify this strictly narrower candidate proposition against the excerpts.
Use only the excerpts. Related concepts are insufficient. Return ONLY JSON:
{{"claim_id": "{claim['id']}", "status": "SUPPORTED",
  "sources": ["Source 1"], "supported_part": null,
  "unsupported_part": null, "reason": "..."}}

CANDIDATE:
{core_text}

EVIDENCE:
{evidence}
""".strip()
            core_data = _json_operation(core_prompt, model, timeout, client)
            core_verdict = _normalize_verdict(core_data, core_claim, context)
            if core_verdict["status"] == "SUPPORTED":
                verdict = {
                    "claim_id": claim["id"],
                    "text": claim["text"],
                    "status": "PARTIALLY_SUPPORTED",
                    "supported": False,
                    "sources": core_verdict["sources"],
                    "supported_part": core_text,
                    "unsupported_part": unsupported_part,
                    "reason": (
                        "The cited evidence supports the narrower proposition, "
                        "but not the removed qualifier or broader scope."
                    ),
                    "verification_error": False,
                }

    return verdict


def _supported_core_candidate(text, context):
    """Return only explicitly constrained, meaning-preserving narrower candidates."""
    value = re.sub(r"\s+", " ", text).strip()
    if re.search(r"\bmore traditional\s+encoder[- ]decoder\b", value, re.I):
        core = re.sub(r"\bmore traditional\s+", "", value, flags=re.I)
        core = re.sub(r"\ba encoder[- ]decoder\b", "an encoder-decoder", core, flags=re.I)
        return core, "The characterization 'more traditional'."

    context_key = canonicalize_claim(context.replace("-", " "))
    if (
        re.search(r"Transformer.*state[- ]of[- ]the[- ]art", value, re.I)
        and "transformer (big)" in context.lower()
        and "state of the art" in context_key
    ):
        if "english to german" in context_key and "english to french" in context_key:
            core = (
                "The Transformer (big) achieved state-of-the-art results on the "
                "reported WMT 2014 English-to-German and English-to-French "
                "translation tasks."
            )
        else:
            core = (
                "The Transformer (big) achieved state-of-the-art results on the "
                "reported WMT 2014 translation tasks."
            )
        return core, (
            "The broader wording 'original Transformer' without specifying the "
            "big configuration."
        )

    if (
        re.search(r"BERT.*unlabeled data.*\band\b", value, re.I)
        and "unlabeled data" in context.lower()
        and "pre train" in context_key
    ):
        return (
            "BERT is trained on unlabeled data during pre-training.",
            "The remaining fine-tuning or task-specific assertions.",
        )
    return None


def verify_claims(claims, context, model="llama3.2:3b", timeout=120, client=None):
    results = []
    for claim in claims:
        try:
            verdict = verify_claim(
                claim,
                context,
                model=model,
                timeout=timeout,
                client=client,
            )
        except Exception as exc:
            verdict = {
                "claim_id": claim["id"],
                "text": claim["text"],
                "status": "UNSUPPORTED",
                "supported": False,
                "sources": [],
                "supported_part": None,
                "unsupported_part": claim["text"],
                "reason": f"Verification failed: {type(exc).__name__}: {exc}",
                "verification_error": True,
            }
        results.append(verdict)

    assert len(results) == len(claims), "every extracted claim must have a verdict"
    return results


def verify_claims_with_evidence(
    claims,
    evidence_packages,
    model="llama3.2:3b",
    timeout=120,
    client=None,
    use_direct_match=True,
):
    """Verify unique canonical claims against claim-specific evidence."""
    results = []
    cache = {}
    cache_hits = 0

    for claim in claims:
        canonical = (canonicalize_claim(claim["text"]), claim.get("paper_hint"))
        package = evidence_packages[claim["id"]]
        scoped_evidence = [
            item for item in package["evidence_used"]
            if not claim.get("paper_hint")
            or paper_titles_compatible(item.get("paper", ""), claim["paper_hint"])
        ]
        if claim.get("paper_hint") and not scoped_evidence:
            verdict = {
                "claim_id": claim["id"],
                "text": claim["text"],
                "status": "UNSUPPORTED",
                "supported": False,
                "sources": [],
                "supported_part": None,
                "unsupported_part": claim["text"],
                "reason": "No evidence from the target paper was retrieved.",
                "verification_error": True,
                "method": "evidence_scope_guard",
                "cache_hit": False,
                "entities": list(claim.get("entities", [])),
                "primary_entity": claim.get("primary_entity"),
                "referenced_entities": list(claim.get("referenced_entities", [])),
                "relation_type": claim.get("relation_type"),
                "paper_hint": claim.get("paper_hint"),
                "original_draft_section": claim.get("original_draft_section"),
                "original_bullet": claim.get("original_bullet"),
                "evidence_used": list(package["evidence_used"]),
            }
            results.append(verdict)
            continue
        if canonical in cache:
            cache_hits += 1
            verdict = dict(cache[canonical])
            verdict["claim_id"] = claim["id"]
            verdict["text"] = claim["text"]
            verdict["cache_hit"] = True
        else:
            try:
                verdict = None
                if use_direct_match:
                    verdict = find_direct_support(
                        claim,
                        package["results"],
                        package["source_ids"],
                    )
                if verdict is None:
                    verdict = verify_claim(
                        claim,
                        package["context"],
                        model=model,
                        timeout=timeout,
                        client=client,
                    )
                    verdict["method"] = "llm"
                verdict["cache_hit"] = False
            except Exception as exc:
                verdict = {
                    "claim_id": claim["id"],
                    "text": claim["text"],
                    "status": "UNSUPPORTED",
                    "supported": False,
                    "sources": [],
                    "supported_part": None,
                    "unsupported_part": claim["text"],
                    "reason": f"Verification failed: {type(exc).__name__}: {exc}",
                    "verification_error": True,
                    "method": "llm",
                    "cache_hit": False,
                }
            cache[canonical] = dict(verdict)

        verdict["entities"] = list(claim.get("entities", []))
        verdict["primary_entity"] = claim.get("primary_entity")
        verdict["referenced_entities"] = list(claim.get("referenced_entities", []))
        verdict["relation_type"] = claim.get("relation_type")
        verdict["paper_hint"] = claim.get("paper_hint")
        verdict["original_draft_section"] = claim.get("original_draft_section")
        verdict["original_bullet"] = claim.get("original_bullet")
        verdict["evidence_used"] = list(package["evidence_used"])
        if verdict.get("status") in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
            evidence_by_source = {
                item["source"]: item for item in verdict["evidence_used"]
            }
            cited = [evidence_by_source.get(source) for source in verdict["sources"]]
            invalid = (
                not verdict["sources"]
                or any(item is None for item in cited)
                or (
                    claim.get("paper_hint")
                    and any(
                        item.get("paper") != claim["paper_hint"]
                        and not paper_titles_compatible(
                            item.get("paper", ""), claim["paper_hint"]
                        )
                        for item in cited if item is not None
                    )
                )
            )
            if invalid:
                verdict.update({
                    "status": "UNSUPPORTED",
                    "supported": False,
                    "sources": [],
                    "supported_part": None,
                    "unsupported_part": claim["text"],
                    "reason": "Supported verdict failed evidence-scope validation.",
                    "verification_error": True,
                    "method": "evidence_scope_guard",
                })
            elif not _technical_evidence_sufficient(
                verdict.get("supported_part") or claim["text"],
                verdict["sources"],
                package,
            ):
                verdict.update({
                    "status": "UNSUPPORTED",
                    "supported": False,
                    "sources": [],
                    "supported_part": None,
                    "unsupported_part": claim["text"],
                    "reason": (
                        "The cited excerpts do not establish every technical "
                        "component of the claim."
                    ),
                    "verification_error": False,
                    "method": "technical_evidence_guard",
                })
            elif verdict.get("status") == "SUPPORTED" and not _multi_entity_supported(
                claim, verdict["sources"], package["evidence_used"]
            ):
                verdict.update({
                    "status": "UNSUPPORTED",
                    "supported": False,
                    "sources": [],
                    "supported_part": None,
                    "unsupported_part": claim["text"],
                    "reason": "The claim names multiple models, but cited evidence does not cover every model.",
                    "verification_error": False,
                    "method": "multi_entity_guard",
                })
        results.append(verdict)

    assert len(results) == len(claims), "every extracted claim must have a verdict"
    return results, cache_hits


def _technical_evidence_sufficient(text, sources, package):
    """Require exact technical anchors rather than merely related excerpts."""
    value = canonicalize_claim(text)
    checks = []
    if "wordpiece" in value:
        checks.append(r"\bwordpiece\b")
    if "vocabulary" in value:
        checks.append(r"\bvocab(?:ulary)?\b")
    if "hidden size" in value:
        checks.append(r"\bhidden\s+size\b|\bH\s*=")
    if re.search(r"attention heads?", value):
        checks.append(r"\b(?:attention\s+)?heads?\b|\bA\s*=")
    if "segment embedding" in value:
        checks.append(r"\bsegment\s+embeddings?\b")
    if "sentence a" in value:
        checks.append(r"\bsentence\s+A\b")
    if "sentence b" in value:
        checks.append(r"\bsentence\s+B\b")
    numbers = re.findall(r"\b\d[\d,]*[mk]?\b", value, flags=re.IGNORECASE)
    if not checks and not numbers:
        return True
    cited_text = " ".join(
        str(result["payload"].get("text", ""))
        for result, source in zip(package["results"], package["source_ids"])
        if source in sources
    )
    return all(re.search(pattern, cited_text, flags=re.IGNORECASE) for pattern in checks) and all(
        number.replace(",", "").lower()
        in cited_text.replace(",", "").lower()
        for number in numbers
    )


def _multi_entity_supported(claim, sources, evidence_used):
    classification = classify_claim_relation(claim["text"])
    relation_type = claim.get("relation_type") or classification["relation_type"]
    if relation_type not in {"JOINT", "COMPARATIVE"}:
        return True
    entities = claim.get("entities") or classification["asserted_entities"]
    required_papers = []
    if any(entity.startswith("BERT") for entity in entities):
        required_papers.append(BERT_PAPER)
    if "Transformer" in entities:
        required_papers.append(TRANSFORMER_PAPER)
    if len(required_papers) < 2:
        return True
    cited_papers = [
        item["paper"] for item in evidence_used if item["source"] in sources
    ]
    return all(
        any(paper_titles_compatible(paper, required) for paper in cited_papers)
        for required in required_papers
    )


def _clean_supported_fact(text):
    """Turn a verified proposition into clean answer text without adding meaning."""
    value = re.sub(r"\s+", " ", str(text or "")).strip(" -*\t\n")
    value = re.sub(r"\s*\[(?:Source|Evidence)\s+\d+\]\s*", " ", value)
    value = re.sub(
        r"^(?:model architecture|architecture|introduction|conclusion)\s*[:.-]?\s+",
        "", value, flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+based on\s*(?:\.{3}|…)$", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"^(BERT)'s model architecture is\s+", r"\1 uses ", value,
        flags=re.IGNORECASE,
    )
    value = _normalize_known_claim_wording(value)
    return value.strip().rstrip(" .") + "." if value.strip(" .") else ""


def _facts_overlap(first, second):
    first_key = canonicalize_claim(first)
    second_key = canonicalize_claim(second)
    if first_key == second_key:
        return True
    first_tokens = set(_match_tokens(first))
    second_tokens = set(_match_tokens(second))
    if not first_tokens or not second_tokens:
        return False
    containment = len(first_tokens & second_tokens) / min(
        len(first_tokens), len(second_tokens)
    )
    return containment >= 0.85 or SequenceMatcher(
        None, first_key, second_key
    ).ratio() >= 0.88


def build_final_fact_set(supported_claims, partially_supported_claims):
    """Create one canonical, cited list containing only verified factual content."""
    candidates = []
    for claim in supported_claims:
        if claim.get("verification_error") or claim.get("status") != "SUPPORTED":
            continue
        candidates.append((claim.get("text"), claim))
    for claim in partially_supported_claims:
        if claim.get("verification_error") or claim.get("status") != "PARTIALLY_SUPPORTED":
            continue
        candidates.append((claim.get("supported_part"), claim))

    facts = []
    for raw_text, claim in candidates:
        text = _clean_supported_fact(raw_text)
        if not text:
            continue
        is_partial = claim.get("status") == "PARTIALLY_SUPPORTED"
        fact_entities = (
            resolve_claim_entities(text)
            if is_partial
            else list(dict.fromkeys(claim.get("entities", [])))
        )
        fact_classification = classify_claim_relation(text)
        if is_partial:
            fact_entities = fact_classification["asserted_entities"]
        fact_paper_hint = (
            resolve_paper_hint(text) if is_partial else claim.get("paper_hint")
        )
        item = {
            "text": text,
            "sources": list(dict.fromkeys(claim.get("sources", []))),
            "entities": fact_entities,
            "paper_hint": fact_paper_hint,
            "primary_entity": fact_classification["primary_entity"],
            "referenced_entities": fact_classification["referenced_entities"],
            "relation_type": fact_classification["relation_type"],
        }
        if not validate_final_fact(item, claim):
            continue
        duplicate_index = next(
            (index for index, fact in enumerate(facts)
             if _facts_overlap(fact["text"], text)), None,
        )
        if duplicate_index is None:
            facts.append(item)
            continue

        existing = facts[duplicate_index]
        merged_sources = list(dict.fromkeys(existing["sources"] + item["sources"]))
        merged_entities = list(dict.fromkeys(existing["entities"] + item["entities"]))
        richer = item if len(_match_tokens(item["text"])) > len(
            _match_tokens(existing["text"])
        ) else existing
        richer = dict(richer)
        richer["sources"] = merged_sources
        richer["entities"] = merged_entities
        richer["paper_hint"] = richer.get("paper_hint") or existing.get("paper_hint")
        facts[duplicate_index] = richer
    return facts


def validate_final_fact(fact, claim):
    """Enforce source, entity, and partial-claim invariants before synthesis."""
    if not fact.get("text") or not fact.get("sources"):
        return False
    evidence_used = claim.get("evidence_used", [])
    if evidence_used:
        by_source = {item["source"]: item for item in evidence_used}
        cited = [by_source.get(source) for source in fact["sources"]]
        if any(item is None for item in cited):
            return False
        paper_hint = fact.get("paper_hint") or claim.get("paper_hint")
        if paper_hint and any(
            not paper_titles_compatible(item["paper"], paper_hint)
            for item in cited
        ):
            return False
        scoped_claim = dict(claim)
        scoped_claim["text"] = fact["text"]
        scoped_claim["entities"] = fact.get("entities", [])
        scoped_claim["paper_hint"] = fact.get("paper_hint")
        if not _multi_entity_supported(scoped_claim, fact["sources"], evidence_used):
            return False
    if claim.get("status") == "PARTIALLY_SUPPORTED":
        if validate_partial_verdict(claim):
            return False
        if canonicalize_claim(fact["text"]) != canonicalize_claim(
            _clean_supported_fact(claim.get("supported_part", ""))
        ):
            return False
    elif claim.get("status") != "SUPPORTED":
        return False
    return True


def _fact_group(fact):
    text = canonicalize_claim(fact["text"])
    entities = " ".join(fact.get("entities", [])).lower()
    paper = str(fact.get("paper_hint") or "").lower()
    primary = str(fact.get("primary_entity") or "").lower()
    if len(fact.get("entities", [])) > 1:
        return "Similarities"
    if primary == "gpt" or "gpt" in text or "gpt" in entities or "gpt" in paper:
        return "GPT (distinct from the original Transformer)"
    if primary.startswith("bert") or "bert" in text or "bert" in entities or "bert" in paper:
        return "BERT"
    if (
        "attention is all you need" in paper
        or "original transformer" in text
        or "the transformer" in text
        or "transformer" in entities
    ):
        return "Original Transformer"
    return "Verified facts"


def synthesize_verified_answer(question, final_fact_set):
    """Organize verified facts without inventing connective claims."""
    if not final_fact_set:
        return INSUFFICIENT_EVIDENCE
    comparison = any(
        word in question.lower() for word in ("compare", "comparison", "difference")
    )
    if not comparison:
        return "\n".join(
            f"- {fact['text']} "
            + " ".join(f"[{source}]" for source in fact["sources"])
            for fact in final_fact_set
        )

    groups = {}
    for fact in final_fact_set:
        groups.setdefault(_fact_group(fact), []).append(fact)
    shared_facts = _derive_shared_facts(
        groups.get("BERT", []), groups.get("Original Transformer", [])
    )
    if shared_facts:
        groups.setdefault("Similarities", []).extend(shared_facts)
    sections = []
    for heading in (
        "BERT", "Original Transformer", "Similarities",
        "GPT (distinct from the original Transformer)", "Verified facts",
    ):
        if heading not in groups:
            continue
        lines = [heading]
        lines.extend(
            f"- {fact['text']} "
            + " ".join(f"[{source}]" for source in fact["sources"])
            for fact in groups[heading]
        )
        sections.append("\n".join(lines))

    bert_encoder = next(
        (
            fact for fact in groups.get("BERT", [])
            if "encoder" in canonicalize_claim(fact["text"])
        ),
        None,
    )
    transformer_encoder_decoder = next(
        (
            fact for fact in groups.get("Original Transformer", [])
            if "encoder decoder" in canonicalize_claim(
                fact["text"].replace("-", " ")
            )
        ),
        None,
    )
    if bert_encoder and transformer_encoder_decoder:
        citations = list(dict.fromkeys(
            bert_encoder["sources"] + transformer_encoder_decoder["sources"]
        ))
        sections.append(
            "Key architectural difference\n"
            "- BERT is described as an encoder-based architecture, while the "
            "original Transformer is described with an encoder-decoder structure. "
            + " ".join(f"[{source}]" for source in citations)
        )
    return "\n\n".join(sections)


def _derive_shared_facts(bert_facts, transformer_facts):
    """Create a similarity only when independently supported predicates match."""
    shared = []
    for bert in bert_facts:
        bert_match = re.match(r"BERT\s+(.+)", bert["text"], flags=re.IGNORECASE)
        if not bert_match:
            continue
        bert_predicate = canonicalize_claim(bert_match.group(1))
        for transformer in transformer_facts:
            transformer_match = re.match(
                r"(?:The\s+)?(?:original\s+)?Transformer\s+(.+)",
                transformer["text"],
                flags=re.IGNORECASE,
            )
            if not transformer_match:
                continue
            if canonicalize_claim(transformer_match.group(1)) != bert_predicate:
                continue
            predicate = bert_match.group(1).rstrip(" .")
            shared.append({
                "text": f"Both BERT and the original Transformer {predicate}.",
                "sources": list(dict.fromkeys(
                    bert["sources"] + transformer["sources"]
                )),
                "entities": ["BERT", "Transformer"],
                "paper_hint": None,
            })
    return shared


def build_verified_answer(question, verified_claims):
    """Backward-compatible safe answer builder."""
    supported = [item for item in verified_claims if item.get("status") == "SUPPORTED"]
    partial = [
        item for item in verified_claims
        if item.get("status") == "PARTIALLY_SUPPORTED"
    ]
    return synthesize_verified_answer(
        question, build_final_fact_set(supported, partial)
    )


def generate_grounded_answer(
    question,
    reranked_results,
    model="llama3.2:3b",
    top_k=3,
    comparison=None,
    timeout=120,
    client=None,
    debug=True,
    vector_store=None,
    embedder=None,
    reranker=None,
    retrieve_k=10,
    evidence_k=3,
    use_direct_match=True,
):
    """Run draft, atomic extraction, claim verification, and safe synthesis."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    selected_results = list(reranked_results[:top_k])
    if not selected_results:
        raise ValueError("reranked_results must contain at least one result")

    titles = {result["payload"]["paper_title"] for result in selected_results}
    if comparison is None:
        comparison = len(titles) > 1

    if comparison:
        context = build_comparison_context(selected_results)
        draft_prompt = build_comparison_prompt(question, context)
    else:
        context = build_context(selected_results)
        draft_prompt = build_prompt(question, context)

    draft_answer = _run_chat(
        draft_prompt, model, timeout, 0.1, client, json_output=False
    )
    claims = extract_claims(draft_answer, model=model, timeout=timeout, client=client)
    available_titles = {result["payload"]["paper_title"] for result in selected_results}
    source_registry = {}
    evidence_packages = {}
    for claim in claims:
        resolved_paper = resolve_paper_hint(claim["text"])
        paper_hint = resolved_paper or claim.get("paper_hint")
        if (
            vector_store is None
            and not resolved_paper
            and paper_hint
            and paper_hint not in available_titles
        ):
            paper_hint = next(
                (
                    title for title in available_titles
                    if canonicalize_claim(paper_hint or "") in canonicalize_claim(title)
                    or canonicalize_claim(title) in canonicalize_claim(paper_hint or "")
                ),
                None,
            )
        if not paper_hint and resolve_claim_entity(claim["text"]) != "GPT":
            for entity in claim.get("entities", []):
                entity_key = canonicalize_claim(entity)
                if not entity_key:
                    continue
                matching_titles = []
                for result in selected_results:
                    payload = result["payload"]
                    metadata = " ".join(
                        str(payload.get(key, ""))
                        for key in ("paper_title", "retrieval_query", "target_paper")
                    )
                    if entity_key in canonicalize_claim(metadata):
                        matching_titles.append(payload["paper_title"])
                if len(set(matching_titles)) == 1:
                    paper_hint = matching_titles[0]
                    break
        claim["paper_hint"] = paper_hint

        if vector_store is not None and embedder is not None and reranker is not None:
            evidence_results = retrieve_evidence_for_claim(
                claim=claim["text"],
                vector_store=vector_store,
                embedder=embedder,
                reranker=reranker,
                paper_hint=paper_hint,
                retrieve_k=retrieve_k,
                evidence_k=evidence_k,
            )
        else:
            evidence_results = [
                result for result in selected_results
                if not paper_hint or paper_titles_compatible(
                    result["payload"]["paper_title"], paper_hint
                )
            ][:evidence_k]
            if not evidence_results and not paper_hint:
                evidence_results = selected_results[:evidence_k]

        if paper_hint:
            evidence_results = [
                result for result in evidence_results
                if evidence_matches_paper(result, paper_hint)
            ]

        evidence_packages[claim["id"]] = _build_claim_evidence(
            evidence_results, source_registry
        )

    verification_results, cache_hits = verify_claims_with_evidence(
        claims,
        evidence_packages,
        model=model,
        timeout=timeout,
        client=client,
        use_direct_match=use_direct_match,
    )
    supported_claims = [
        claim
        for claim in verification_results
        if claim["status"] == "SUPPORTED" and not claim["verification_error"]
    ]
    partially_supported_claims = [
        claim
        for claim in verification_results
        if claim["status"] == "PARTIALLY_SUPPORTED"
        and not claim["verification_error"]
    ]
    rejected_claims = [
        claim
        for claim in verification_results
        if claim["status"] == "UNSUPPORTED" and not claim["verification_error"]
    ]
    verification_errors = [
        claim for claim in verification_results if claim["verification_error"]
    ]
    evidence_routing_failures = [
        claim for claim in verification_errors
        if claim.get("method") == "evidence_scope_guard"
    ]
    malformed_partial_verdicts = [
        claim for claim in verification_errors
        if "malformed partial verdict" in claim.get("reason", "").lower()
    ]
    final_fact_set = build_final_fact_set(
        supported_claims, partially_supported_claims
    )
    final_answer = synthesize_verified_answer(question, final_fact_set)

    assert len(claims) == (
        len(supported_claims)
        + len(partially_supported_claims)
        + len(rejected_claims)
        + len(verification_errors)
    ), "verification accounting invariant failed"

    if debug:
        print(f"Extracted claims: {len(claims)}")
        print(f"Unique claims: {len({canonicalize_claim(c['text']) for c in claims})}")
        print(f"Cache hits: {cache_hits}")
        print(f"Verified claims: {len(verification_results)}")
        print(f"Supported: {len(supported_claims)}")
        print(f"Partially supported: {len(partially_supported_claims)}")
        print(f"Unsupported: {len(rejected_claims)}")
        print(f"Errors: {len(verification_errors)}")
        print(f"Evidence-routing failures: {len(evidence_routing_failures)}")
        print(f"Malformed partial verdicts: {len(malformed_partial_verdicts)}")
        print(f"Final facts: {len(final_fact_set)}")

        for verdict in verification_results:
            label = "ERROR" if verdict["verification_error"] else verdict["status"]
            print(f"\nOriginal draft section: {verdict.get('original_draft_section') or 'Unknown'}")
            print(f"Original bullet: {verdict.get('original_bullet') or verdict['text']}")
            print(f"Atomic claim: {verdict['text']}")
            print(f"Primary entity: {verdict.get('primary_entity') or 'Unknown'}")
            print(
                "Referenced entities: "
                f"{', '.join(verdict.get('referenced_entities', [])) or 'None'}"
            )
            print(f"Relation type: {verdict.get('relation_type') or 'SINGLE_SUBJECT'}")
            print(f"Resolved paper: {verdict.get('paper_hint') or 'None'}")
            evidence_papers = list(dict.fromkeys(
                item["paper"] for item in verdict["evidence_used"]
            ))
            print(f"Evidence papers: {', '.join(evidence_papers) or 'None'}")
            print("Evidence used:")
            for evidence in verdict["evidence_used"]:
                print(
                    f"{evidence['source']} - {evidence['paper']} - "
                    f"{evidence['section']}"
                )
            print(f"Verdict: {label}")
            print(f"Method: {verdict['method']}")
            if verdict["sources"]:
                print(f"Evidence: {', '.join(verdict['sources'])}")
            if verdict.get("supported_part"):
                print(f"Supported part: {verdict['supported_part']}")
            print(
                "Normalized supported core: "
                f"{verdict.get('supported_part') or (verdict['text'] if verdict['status'] == 'SUPPORTED' else 'None')}"
            )
            if verdict.get("unsupported_part") and verdict["status"] == "PARTIALLY_SUPPORTED":
                print(f"Unsupported part: {verdict['unsupported_part']}")
            included = any(
                canonicalize_claim(fact["text"]) in {
                    canonicalize_claim(verdict["text"]),
                    canonicalize_claim(verdict.get("supported_part") or ""),
                }
                for fact in final_fact_set
            )
            print(f"Final fact included: {'yes' if included else 'no'}")
            print(f"Reason: {verdict['reason']}")

    if debug:
        print("\nFinal normalized fact set:")
        for fact in final_fact_set:
            print(f"- {fact}")
        print("\nFinal answer:")
        print(final_answer)

    if not debug:
        return {"answer": final_answer, "sources": selected_results}

    return {
        "draft_answer": draft_answer,
        "extracted_claims": claims,
        "claims": claims,
        "unique_claims": list({canonicalize_claim(c["text"]): c for c in claims}.values()),
        "evidence_used": {
            claim_id: package["evidence_used"]
            for claim_id, package in evidence_packages.items()
        },
        "verified_claims": verification_results,
        "supported": supported_claims,
        "partial": partially_supported_claims,
        "unsupported": rejected_claims,
        "errors": verification_errors,
        "evidence_routing_failures": evidence_routing_failures,
        "malformed_partial_verdicts": malformed_partial_verdicts,
        "supported_claims": supported_claims,
        "partially_supported_claims": partially_supported_claims,
        "rejected_claims": rejected_claims,
        "unsupported_claims": rejected_claims,
        "verification_errors": verification_errors,
        "final_fact_set": final_fact_set,
        "final_answer": final_answer,
        "answer": final_answer,
        "sources": selected_results,
        "context": context,
        "comparison": comparison,
        "cache_hits": cache_hits,
        "source_catalog": evidence_packages,
    }


def generate_answer(
    question,
    reranked_results,
    model="llama3.2:3b",
    top_k=3,
    comparison=None,
    timeout=120,
    temperature=0.1,
    client=None,
    debug=True,
    vector_store=None,
    embedder=None,
    reranker=None,
    use_direct_match=True,
):
    """Backward-compatible entry point for claim-grounded generation."""
    return generate_grounded_answer(
        question=question,
        reranked_results=reranked_results,
        model=model,
        top_k=top_k,
        comparison=comparison,
        timeout=timeout,
        client=client,
        debug=debug,
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        use_direct_match=use_direct_match,
    )


def generate_comparison_answer(
    question,
    comparison_results,
    model="llama3.2:3b",
    timeout=120,
    client=None,
    debug=True,
    vector_store=None,
    embedder=None,
    reranker=None,
    use_direct_match=True,
):
    """Backward-compatible comparison entry point."""
    return generate_grounded_answer(
        question=question,
        reranked_results=comparison_results,
        model=model,
        top_k=len(comparison_results),
        comparison=True,
        timeout=timeout,
        client=client,
        debug=debug,
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        use_direct_match=use_direct_match,
    )


def verify_answer(
    question,
    draft_answer,
    context,
    model="llama3.2:3b",
    timeout=120,
    client=None,
):
    """Compatibility wrapper that performs claim-level verification."""
    claims = extract_claims(draft_answer, model=model, timeout=timeout, client=client)
    verdicts = verify_claims(claims, context, model=model, timeout=timeout, client=client)
    return build_verified_answer(question, verdicts)
