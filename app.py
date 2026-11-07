"""Streamlit interface for the research-paper RAG pipeline."""

import os
from pathlib import Path

import streamlit as st

from src.embeddings.embedder import Embedder
from src.generation.generator import generate_evidence_answer, generate_grounded_answer
from src.generation.groq_client import GroqClient
from src.ingestion.chunker import create_chunks
from src.ingestion.grobid_client import GrobidClient
from src.ingestion.figure_images import figure_data_url
from src.ingestion.grobid_parser import parse_grobid_tei
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PARSED_DIR = ROOT / "data" / "parsed"
COLLECTION_NAME = "research_papers_multi"


st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1320px; padding-top: 2rem;}
      [data-testid="stSidebar"] {background: #f3f7fa;}
      .hero {padding: 1.35rem 1.6rem; border-radius: 18px;
             background: linear-gradient(120deg, #12355b, #2176ae); color: white;}
      .hero h1 {margin: 0; font-size: 2.2rem;}
      .hero p {margin: .45rem 0 0; color: #e9c46a; font-size: 1.05rem;}
      .source-card {border-left: 4px solid #2a9d8f; padding: .2rem .9rem;
                    margin: .5rem 0; background: #f7fbfb;}
      .muted {color: #5d6b78;}
    </style>
    <div class="hero">
      <h1>Research Paper Assistant</h1>
      <p>Local, evidence-first answers across your scientific papers</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return Embedder()


@st.cache_resource(show_spinner="Loading reranker…")
def get_reranker():
    return Reranker()


@st.cache_resource
def get_vector_store():
    return VectorStore(collection_name=COLLECTION_NAME)


def indexed_papers(vector_store):
    """Return unique paper titles currently stored in Qdrant."""
    if not vector_store.client.collection_exists(vector_store.collection_name):
        return []

    titles = set()
    offset = None
    while True:
        points, offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            limit=100,
            offset=offset,
            with_payload=["paper_title"],
            with_vectors=False,
        )
        titles.update(
            point.payload.get("paper_title", "").strip()
            for point in points
            if point.payload and point.payload.get("paper_title")
        )
        if offset is None:
            break
    return sorted(titles, key=str.casefold)


def index_upload(uploaded_file, vector_store, embedder):
    """Persist, parse, chunk, embed, and index one uploaded PDF."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError(f"{safe_name} is not a PDF")

    pdf_path = RAW_DIR / safe_name
    pdf_path.write_bytes(uploaded_file.getbuffer())
    tei_path = PARSED_DIR / f"{pdf_path.stem}.tei.xml"

    if not tei_path.exists():
        GrobidClient().process_pdf(pdf_path, tei_path)

    paper = parse_grobid_tei(tei_path)
    chunks = create_chunks(paper, document_name=safe_name)
    if not chunks:
        raise RuntimeError(f"GROBID produced no indexable text for {safe_name}")

    embeddings = embedder.embed_texts([chunk["text"] for chunk in chunks])
    vector_store.create_collection(vector_size=len(embeddings[0]))
    vector_store.delete_document(safe_name)
    vector_store.add_chunks(chunks, embeddings)
    return paper["title"], len(chunks)


def requested_element_type(question):
    """Infer an explicit structured-evidence request from the user's wording."""
    lowered = question.casefold()
    aliases = {
        "equation": ("equation", "formula"),
        "table": ("table", "tabular"),
        "figure": ("figure", "diagram", "chart", "graph"),
        "reference": ("reference", "bibliography", "cited", "citation"),
    }
    return next(
        (kind for kind, words in aliases.items() if any(word in lowered for word in words)),
        None,
    )


def requested_overview_aspects(question):
    """Return distinct information needs in a broad multi-paper question."""
    lowered = question.casefold()
    aspect_terms = {
        "objectives": ("objective", "purpose", "research problem", "goal"),
        "methods": ("method", "methodology", "approach", "architecture", "training"),
        "datasets": ("dataset", "data set", "corpus", "benchmark"),
        "results": ("result", "performance", "finding", "metric", "score"),
        "limitations": ("limitation", "weakness", "future work"),
        "contributions": ("contribution", "novelty", "propose"),
    }
    return [
        aspect
        for aspect, terms in aspect_terms.items()
        if any(term in lowered for term in terms)
    ]


def _retrieve_multi_paper_overview(
    question, aspects, selected_papers, vector_store, embedder, reranker, limit
):
    """Retrieve one focused evidence passage per aspect and paper."""
    aspect_queries = {
        "objectives": "What research problem, motivation, goal, or objective does the paper state?",
        "methods": "What method, model architecture, training procedure, or experimental approach does the paper use?",
        "datasets": "Which datasets, corpora, benchmarks, and evaluation tasks does the paper use?",
        "results": "What quantitative results, metrics, scores, and main findings does the paper report?",
        "limitations": "What limitations, weaknesses, failure cases, or future work does the paper state?",
        "contributions": "What are the paper's main contributions and proposed innovations?",
    }
    selected = []
    seen = set()
    evidence_per_aspect = {
        "datasets": 2,
        "results": 3,
    }
    for title in selected_papers:
        for aspect in aspects:
            focused_question = f"{title}. {aspect_queries[aspect]}"
            vector = embedder.embed_query(focused_question)
            candidates = vector_store.search_by_paper(vector, title, limit=max(limit, 8))
            ranked = reranker.rerank(focused_question, candidates, top_k=5)
            target_count = evidence_per_aspect.get(aspect, 1)
            choices = [item for item in ranked if _point_key(item) not in seen][
                :target_count
            ]
            for choice in choices:
                choice = dict(choice)
                choice["retrieval_aspect"] = aspect
                selected.append(choice)
                seen.add(_point_key(choice))
    return selected


def _point_key(result):
    payload = result.get("payload", {}) if isinstance(result, dict) else result.payload
    return (
        payload.get("paper_title", ""),
        payload.get("section", ""),
        payload.get("element_type", "paragraph"),
        payload.get("label", ""),
        payload.get("text", ""),
    )


def _keep_requested_element(reranked, typed_candidates, element_type, top_k):
    """Keep the requested element first in the final evidence window."""
    if not typed_candidates:
        return reranked
    matching = [
        item for item in reranked
        if item.get("payload", {}).get("element_type") == element_type
    ]
    if matching:
        selected = matching[0]
        return [selected] + [item for item in reranked if item is not selected][: top_k - 1]
    typed = typed_candidates[0]
    typed_result = {
        "score": getattr(typed, "score", 0.0),
        "original_score": getattr(typed, "score", 0.0),
        "payload": dict(typed.payload),
    }
    return ([typed_result] + reranked)[:top_k]


def retrieve(question, selected_papers, vector_store, embedder, reranker, limit, top_k):
    aspects = requested_overview_aspects(question)
    if selected_papers and len(selected_papers) > 1 and len(aspects) >= 2:
        overview_results = _retrieve_multi_paper_overview(
            question,
            aspects,
            selected_papers,
            vector_store,
            embedder,
            reranker,
            limit,
        )
        if overview_results:
            return overview_results
    query_vector = embedder.embed_query(question)
    element_type = requested_element_type(question)
    typed_candidates = []

    if selected_papers:
        # Rerank within each paper so a comparison cannot be dominated by one title.
        per_paper_top_k = max(1, top_k // len(selected_papers))
        results = []
        for title in selected_papers:
            candidates = vector_store.search_by_paper(
                query_vector, title, limit=max(limit, per_paper_top_k)
            )
            typed_for_paper = []
            if element_type:
                typed_for_paper = list(vector_store.search_by_element_type(
                    query_vector, element_type, limit=3, paper_title=title
                ))
                typed_candidates.extend(typed_for_paper)
                known = {_point_key(item) for item in candidates}
                candidates.extend(item for item in typed_for_paper if _point_key(item) not in known)
            results.extend(
                reranker.rerank(question, candidates, top_k=per_paper_top_k)
            )
        results = sorted(results, key=lambda item: item["score"], reverse=True)
        return _keep_requested_element(results, typed_candidates, element_type, top_k)
    else:
        candidates = list(vector_store.search(query_vector, limit=limit))
        if element_type:
            typed_candidates = list(vector_store.search_by_element_type(
                query_vector, element_type, limit=3
            ))
            known = {_point_key(item) for item in candidates}
            candidates.extend(item for item in typed_candidates if _point_key(item) not in known)

    if not candidates:
        return []
    reranked = reranker.rerank(question, candidates, top_k=top_k)
    return _keep_requested_element(reranked, typed_candidates, element_type, top_k)


def source_label(source, number):
    payload = source.get("payload", {})
    element_type = payload.get("element_type", "paragraph").replace("_", " ").title()
    label = payload.get("label", "")
    evidence_kind = f"{element_type} {label}".strip()
    return (
        f"Source {number} — {payload.get('paper_title', 'Unknown paper')} · "
        f"{payload.get('section', 'Unknown section')} · {evidence_kind}"
    )


def answer_sources(result):
    """Prefer the claim verifier's source registry over initial retrieval order."""
    catalog = result.get("source_catalog", {})
    by_id = {}
    for package in catalog.values():
        for source_id, source in zip(package.get("source_ids", []), package.get("results", [])):
            by_id.setdefault(source_id, source)
    if by_id:
        return [
            (source_id, by_id[source_id])
            for source_id in sorted(by_id, key=lambda value: int(value.split()[-1]))
        ]
    return [
        (f"Source {number}", source)
        for number, source in enumerate(result.get("sources", []), start=1)
    ]


def visual_evidence(results):
    """Render retrieved, coordinate-bearing figures for a vision-capable provider."""
    images = []
    for result in results:
        payload = result.get("payload", {})
        if payload.get("element_type") != "figure" or not payload.get("coords"):
            continue
        image = figure_data_url(RAW_DIR / payload.get("document", ""), payload["coords"])
        if image:
            images.append(image)
    return images[:2]


def render_result(result):
    st.markdown(result["answer"])

    shown_images = set()
    for source_id, source in answer_sources(result):
        payload = source.get("payload", {})
        coords = payload.get("coords", "")
        document = payload.get("document", "")
        if payload.get("element_type") != "figure" or not coords or not document:
            continue
        image_key = (document, coords)
        if image_key in shown_images:
            continue
        image = figure_data_url(RAW_DIR / document, coords)
        if not image:
            continue
        shown_images.add(image_key)
        label = f"Figure {payload.get('label', '')}".strip()
        caption = payload.get("caption", "")
        st.image(
            image,
            caption=f"{label} — {caption} ({source_id})" if caption else f"{label} ({source_id})",
        )

    st.subheader("Sources")
    for source_id, source in answer_sources(result):
        payload = source.get("payload", {})
        number = int(source_id.split()[-1])
        with st.expander(source_label(source, number)):
            score = source.get("score")
            if score is not None:
                st.caption(f"Reranker score: {score:.3f}")
            if payload.get("element_type") == "figure" and payload.get("coords"):
                image = figure_data_url(
                    RAW_DIR / payload.get("document", ""), payload["coords"]
                )
                if image:
                    st.image(image, caption=payload.get("caption") or payload.get("label"))
            st.write(payload.get("text", "No excerpt available."))

    with st.expander("Claim verification", expanded=False):
        verdicts = result.get("verified_claims", [])
        if not verdicts:
            st.info("No claim-level details were returned.")
        for verdict in verdicts:
            status = "ERROR" if verdict.get("verification_error") else verdict.get("status")
            icon = {"SUPPORTED": "✅", "PARTIALLY_SUPPORTED": "🟡", "UNSUPPORTED": "❌"}.get(status, "⚠️")
            st.markdown(f"**{icon} {status}** — {verdict.get('text', '')}")
            if verdict.get("reason"):
                st.caption(verdict["reason"])


with st.sidebar:
    st.header("Paper library")
    uploads = st.file_uploader(
        "Upload scientific PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="GROBID must be running at localhost:8070.",
    )

    if st.button("Process and index", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one PDF first.")
        else:
            try:
                store = get_vector_store()
                embedder = get_embedder()
                progress = st.progress(0, text="Preparing papers…")
                indexed = []
                for position, upload in enumerate(uploads, start=1):
                    progress.progress(
                        (position - 1) / len(uploads),
                        text=f"Processing {upload.name}…",
                    )
                    indexed.append(index_upload(upload, store, embedder))
                progress.progress(1.0, text="Indexing complete")
                st.cache_data.clear()
                for title, count in indexed:
                    st.success(f"Indexed {title} ({count} chunks)")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    try:
        paper_options = indexed_papers(get_vector_store())
        st.caption(f"{len(paper_options)} indexed paper(s)")
    except Exception as exc:
        paper_options = []
        st.warning("Qdrant is unavailable. Start it on localhost:6333.")
        st.caption(str(exc))

    scope = st.multiselect(
        "Search scope",
        options=paper_options,
        placeholder="All indexed papers",
        help="Select one paper for focused QA or several for comparison.",
    )
    if len(scope) > 1:
        st.success(f"Multi-paper mode: {len(scope)} papers selected")
        for selected_title in scope:
            st.caption(f"• {selected_title}")
    elif len(scope) == 1:
        st.warning("Single-paper mode: select at least two papers for comparison.")
        st.caption(f"• {scope[0]}")
    else:
        st.info("Library-wide search: no explicit comparison scope selected.")
    provider = st.selectbox(
        "Generation provider", options=["Ollama (local)", "Groq API"]
    )
    default_model = (
        "llama3.1:8b" if provider == "Ollama (local)" else "qwen/qwen3.8-27b"
    )
    model = st.text_input("Model", value=default_model, key=f"model_{provider}")
    if provider == "Groq API":
        if os.environ.get("GROQ_API_KEY"):
            st.caption("Groq API key detected")
        else:
            st.warning("Set GROQ_API_KEY and restart Streamlit before using Groq.")
    with st.expander("Retrieval settings"):
        candidate_limit = st.slider("Candidates per paper", 3, 20, 8)
        evidence_count = st.slider("Evidence passages", 2, 10, 3)
        answer_mode = st.radio(
            "Answer mode",
            options=["Fast", "Verified"],
            help=(
                "Fast uses one grounded model call. Verified extracts and checks "
                "each claim and can take several minutes on CPU."
            ),
        )
        show_verification = st.toggle(
            "Return claim diagnostics",
            value=True,
            disabled=answer_mode == "Fast",
        )


st.markdown("### Ask your library")
st.caption(
    "Choose papers in the sidebar, or leave the scope empty to search the full library. "
    "Selecting multiple papers enables comparison mode."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("result"):
            render_result(message["result"])
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask a question about the indexed papers…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not paper_options:
            st.warning("Index at least one paper before asking a question.")
        else:
            try:
                with st.status("Finding and verifying evidence…", expanded=True) as status:
                    st.write(f"Provider: {provider} · Model: {model}")
                    st.write(
                        "Scope: "
                        + (
                            " | ".join(scope)
                            if scope
                            else "all indexed papers (library-wide search)"
                        )
                    )
                    st.write("Embedding the question and retrieving passages")
                    store = get_vector_store()
                    embedder = get_embedder()
                    reranker = get_reranker()
                    results = retrieve(
                        question,
                        scope,
                        store,
                        embedder,
                        reranker,
                        candidate_limit,
                        evidence_count,
                    )
                    if not results:
                        raise RuntimeError("No evidence was found in the selected scope")
                    if answer_mode == "Fast":
                        st.write("Generating a citation-grounded answer")
                        generation_client = (
                            GroqClient(timeout=90, images=visual_evidence(results))
                            if provider == "Groq API" else None
                        )
                        answer = generate_evidence_answer(
                            question=question,
                            reranked_results=results,
                            model=model,
                            top_k=len(results),
                            comparison=len(scope) > 1,
                            timeout=90,
                            client=generation_client,
                        )
                    else:
                        st.write("Generating a draft and checking its atomic claims")
                        generation_client = (
                            GroqClient(timeout=90, images=visual_evidence(results))
                            if provider == "Groq API" else None
                        )
                        answer = generate_grounded_answer(
                            question=question,
                            reranked_results=results,
                            model=model,
                            top_k=len(results),
                            comparison=len(scope) > 1,
                            timeout=90,
                            client=generation_client,
                            debug=show_verification,
                            vector_store=store,
                            embedder=embedder,
                            reranker=reranker,
                        )
                    status.update(label="Grounded answer ready", state="complete", expanded=False)

                # debug=False returns fewer fields but always includes answer and sources.
                render_result(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer["answer"], "result": answer}
                )
            except Exception as exc:
                st.error(str(exc))
                st.info(
                    "Check that Qdrant (port 6333), GROBID (port 8070 for uploads), "
                    "and the selected generation provider are configured."
                )

