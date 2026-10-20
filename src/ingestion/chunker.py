def flatten_sections(sections, parent_title=""):
    """
    Convert nested sections into a flat list while preserving hierarchy.
    """

    flattened = []

    for section in sections:
        title = section.get("title", "").strip()

        full_title = title

        if parent_title and title:
            full_title = f"{parent_title} > {title}"
        elif parent_title:
            full_title = parent_title

        flattened.append({
            "section": full_title,
            "paragraphs": section.get("paragraphs", [])
        })

        subsections = section.get("subsections", [])

        if subsections:
            flattened.extend(
                flatten_sections(
                    subsections,
                    parent_title=full_title
                )
            )

    return flattened

def chunk_paragraphs(paragraphs, max_words=600):
    chunks = []

    current_chunk = []
    current_words = 0

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph_words = len(paragraph.split())

        if current_words + paragraph_words > max_words and current_chunk:

            chunks.append(" ".join(current_chunk))

            current_chunk = []
            current_words = 0

        current_chunk.append(paragraph)
        current_words += paragraph_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def create_chunks(paper, document_name):
    chunks = []

    chunk_id = 0

    sections = flatten_sections(
        paper["sections"]
    )

    for section in sections:

        section_chunks = chunk_paragraphs(
            section["paragraphs"]
        )

        for text in section_chunks:

            chunks.append({
                "chunk_id": chunk_id,
                "document": document_name,
                "paper_title": paper["title"],
                "section": section["section"],
                "text": text
            })

            chunk_id += 1

    return chunks