from pathlib import Path
import xml.etree.ElementTree as ET
import re


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _attribute(element, name):
    """Return a TEI attribute, including attributes in the XML namespace."""
    if element is None:
        return ""
    return element.get(name, element.get(f"{{http://www.w3.org/XML/1998/namespace}}{name}", ""))


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


def _table_text(table):
    """Convert a TEI table to compact, readable rows for embedding and display."""
    rows = []
    for row in table.findall(".//tei:row", TEI_NS):
        cells = [get_text(cell) for cell in row.findall("tei:cell", TEI_NS)]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows) or get_text(table)


def extract_section_elements(div, section_title):
    """Extract non-paragraph scientific evidence directly owned by a section."""
    elements = []
    for child in list(div):
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "figure":
            kind = "table" if child.get("type", "").lower() == "table" or child.find("tei:table", TEI_NS) is not None else "figure"
            label = get_text(child.find("tei:label", TEI_NS))
            caption = get_text(child.find("tei:head", TEI_NS)) or get_text(child.find("tei:figDesc", TEI_NS))
            table = child.find("tei:table", TEI_NS)
            graphic = child.find("tei:graphic", TEI_NS)
            coords = graphic.get("coords", "") if graphic is not None else ""
            content = _table_text(table) if table is not None else get_text(child.find("tei:figDesc", TEI_NS))
            text = ". ".join(part for part in (f"{kind.title()} {label}".strip(), caption, content) if part)
            if text:
                elements.append({"type": kind, "label": label, "caption": caption, "text": text, "section": section_title, "coords": coords})
        elif tag == "formula":
            label = get_text(child.find("tei:label", TEI_NS)) or _attribute(child, "id")
            formula = get_text(child)
            if formula:
                elements.append({"type": "equation", "label": label, "caption": "", "text": f"Equation {label}: {formula}".strip(), "section": section_title})
    return elements


def extract_body_elements(root):
    """Extract figures/tables placed directly under TEI body, outside sections."""
    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        return []
    wrapper = ET.Element(f"{{{TEI_NS['tei']}}}div")
    for child in list(body):
        if child.tag.rsplit("}", 1)[-1] in {"figure", "formula"}:
            wrapper.append(child)
    return extract_section_elements(wrapper, "Figures and Tables")


def extract_references(root):
    """Extract bibliography entries as independently retrievable evidence."""
    references = []
    for number, entry in enumerate(root.findall(".//tei:text/tei:back//tei:listBibl/*", TEI_NS), 1):
        text = get_text(entry)
        if not text:
            continue
        reference_id = _attribute(entry, "id") or str(number)
        references.append({
            "type": "reference",
            "label": reference_id,
            "caption": "",
            "text": f"Reference {reference_id}: {text}",
            "section": "References",
        })
    return references


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
        "elements": extract_section_elements(div, full_title or "Body"),
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
        "elements": extract_body_elements(root),
        "references": extract_references(root),
        "source_file": tei_path.name
    }

    return document
