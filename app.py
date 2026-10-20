"""Streamlit interface for the research-paper RAG pipeline."""

from pathlib import Path

import streamlit as st

from src.embeddings.embedder import Embedder
from src.generation.generator import generate_evidence_answer, generate_grounded_answer
from src.ingestion.chunker import create_chunks
from src.ingestion.grobid_client import GrobidClient
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
    vector_store.add_chunks(chunks, embeddings)
    return paper["title"], len(chunks)


def retrieve(question, selected_papers, vector_store, embedder, reranker, limit, top_k):
    query_vector = embedder.embed_query(question)

    if selected_papers:
        # Rerank within each paper so a comparison cannot be dominated by one title.
        per_paper_top_k = max(1, top_k // len(selected_papers))
        results = []
        for title in selected_papers:
            candidates = vector_store.search_by_paper(
                query_vector, title, limit=max(limit, per_paper_top_k)
            )
            results.extend(
                reranker.rerank(question, candidates, top_k=per_paper_top_k)
            )
        return sorted(results, key=lambda item: item["score"], reverse=True)
    else:
        candidates = vector_store.search(query_vector, limit=limit)

    if not candidates:
        return []
    return reranker.rerank(question, candidates, top_k=top_k)


def source_label(source, number):
    payload = source.get("payload", {})
    return (
        f"Source {number} — {payload.get('paper_title', 'Unknown paper')} · "
        f"{payload.get('section', 'Unknown section')}"
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


def render_result(result):
    st.markdown(result["answer"])

    st.subheader("Sources")
    for source_id, source in answer_sources(result):
        payload = source.get("payload", {})
        number = int(source_id.split()[-1])
        with st.expander(source_label(source, number)):
            score = source.get("score")
            if score is not None:
                st.caption(f"Reranker score: {score:.3f}")
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
    model = st.text_input("Ollama model", value="llama3.2:3b")
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
                        answer = generate_evidence_answer(
                            question=question,
                            reranked_results=results,
                            model=model,
                            top_k=len(results),
                            comparison=len(scope) > 1,
                            timeout=90,
                        )
                    else:
                        st.write("Generating a draft and checking its atomic claims")
                        answer = generate_grounded_answer(
                            question=question,
                            reranked_results=results,
                            model=model,
                            top_k=len(results),
                            comparison=len(scope) > 1,
                            timeout=90,
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
                    "and Ollama are running, and that the selected Ollama model is installed."
                )

