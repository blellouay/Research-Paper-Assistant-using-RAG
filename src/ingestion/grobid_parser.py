from pathlib import Path
import xml.etree.ElementTree as ET
import re


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def get_text(element):
    """
    Extract all text from an XML element, including nested tags.
    """
    if element is None:
        return ""

    text = " ".join(element.itertext())

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_title(title: str) -> str:
    """
    Clean common cases where GROBID merges legal / permission text
    with the real paper title.

    This is intentionally generic and does not depend on a specific paper.
    """

    if not title:
        return ""

    title = re.sub(r"\s+", " ", title).strip()

    # Common legal / permission phrases that can appear before the title
    legal_markers = [
        "provided proper attribution",
        "permission to reproduce",
        "solely for use in journalistic or scholarly works",
        "all rights reserved",
        "copyright",
    ]

    lowered = title.lower()

    # If the beginning looks like legal boilerplate, try to keep
    # the text after its final sentence.
    if any(marker in lowered for marker in legal_markers):
        sentences = re.split(r"(?<=[.!?])\s+", title)

        # Look for the first sentence after the legal-looking text
        non_legal_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if any(marker in sentence_lower for marker in legal_markers):
                continue

            if sentence.strip():
                non_legal_sentences.append(sentence.strip())

        if non_legal_sentences:
            return " ".join(non_legal_sentences)

    return title


def extract_title(root):
    """
    Extract and normalize the main title.
    """

    title_element = root.find(
        ".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title",
        TEI_NS
    )

    raw_title = get_text(title_element)

    return normalize_title(raw_title)


def extract_abstract(root):
    """
    Extract the abstract.
    """

    abstract_element = root.find(
        ".//tei:teiHeader/tei:profileDesc/tei:abstract",
        TEI_NS
    )

    return get_text(abstract_element)


def extract_authors(root):
    """
    Extract authors from GROBID metadata.

    Note:
    GROBID author parsing is not always perfect, so this should be
    treated as metadata rather than trusted scientific content.
    """

    authors = []

    author_elements = root.findall(
        ".//tei:teiHeader/tei:fileDesc/"
        "tei:sourceDesc/tei:biblStruct/"
        "tei:analytic/tei:author",
        TEI_NS
    )

    for author in author_elements:
        pers_name = author.find("tei:persName", TEI_NS)

        if pers_name is None:
            continue

        first_names = [
            get_text(element)
            for element in pers_name.findall("tei:forename", TEI_NS)
        ]

        surname_element = pers_name.find("tei:surname", TEI_NS)
        surname = get_text(surname_element)

        name_parts = [name for name in first_names if name]

        if surname:
            name_parts.append(surname)

        full_name = " ".join(name_parts).strip()

        if full_name:
            authors.append(full_name)

    return authors


def parse_section(div, parent_title=""):
    """
    Recursively parse a TEI <div> section and its subsections.
    """

    heading_element = div.find("tei:head", TEI_NS)
    heading = get_text(heading_element)

    if parent_title and heading:
        full_title = f"{parent_title} > {heading}"
    elif heading:
        full_title = heading
    else:
        full_title = parent_title

    paragraphs = []

    # Only direct paragraphs belonging to this section.
    # This avoids duplicating paragraphs from subsections.
    for paragraph in div.findall("tei:p", TEI_NS):
        paragraph_text = get_text(paragraph)

        if paragraph_text:
            paragraphs.append(paragraph_text)

    subsections = []

    for child_div in div.findall("tei:div", TEI_NS):
        subsections.append(
            parse_section(
                child_div,
                parent_title=full_title
            )
        )

    return {
        "title": heading,
        "full_title": full_title,
        "paragraphs": paragraphs,
        "subsections": subsections
    }


def extract_sections(root):
    """
    Extract all body sections recursively.
    """

    sections = []

    body = root.find(".//tei:text/tei:body", TEI_NS)

    if body is None:
        return sections

    for div in body.findall("tei:div", TEI_NS):
        sections.append(
            parse_section(div)
        )

    return sections


def parse_grobid_tei(tei_path):
    """
    Parse a GROBID TEI XML file into a structured Python dictionary.
    """

    tei_path = Path(tei_path)

    tree = ET.parse(tei_path)
    root = tree.getroot()

    document = {
        "title": extract_title(root),
        "authors": extract_authors(root),
        "abstract": extract_abstract(root),
        "sections": extract_sections(root),
        "source_file": tei_path.name
    }

    return document