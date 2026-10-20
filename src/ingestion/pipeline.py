from pathlib import Path

from src.ingestion.grobid_client import GrobidClient
from src.ingestion.grobid_parser import parse_grobid_tei
from src.ingestion.chunker import create_chunks


def ingest_papers(
    raw_dir,
    parsed_dir
):
    raw_dir = Path(raw_dir)
    parsed_dir = Path(parsed_dir)

    grobid = GrobidClient()

    all_chunks = []
    papers = []

    pdf_files = list(raw_dir.glob("*.pdf"))

    print(f"Found {len(pdf_files)} papers.")

    for pdf_path in pdf_files:

        print()
        print("=" * 80)
        print("Processing:", pdf_path.name)

        # -------------------------
        # GROBID
        # -------------------------

        tei_path = parsed_dir / (
            pdf_path.stem + ".tei.xml"
        )

        if not tei_path.exists():

            print("Sending to GROBID...")

            grobid.process_pdf(
                pdf_path,
                tei_path
            )

        else:
            print("TEI already exists.")

        # -------------------------
        # Parse TEI
        # -------------------------

        paper = parse_grobid_tei(
            tei_path
        )

        print("Title:", paper["title"])

        papers.append(paper)

        # -------------------------
        # Chunk
        # -------------------------

        chunks = create_chunks(
            paper,
            document_name=pdf_path.name
        )

        print("Chunks:", len(chunks))

        all_chunks.extend(chunks)

    return papers, all_chunks