import re


def clean_text(text: str) -> str:
    """
    Basic cleaning for research-paper text extracted from PDFs.
    """

    # Remove the Google permission notice from Attention Is All You Need
    text = re.sub(
        r"Provided proper attribution is provided, Google hereby grants permission to\s+"
        r"reproduce the tables and figures in this paper solely for use in journalistic or\s+"
        r"scholarly works\.",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Join words split by hyphen + newline
    # Example:
    # "position-\nwise" -> "positionwise"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace ordinary line breaks inside paragraphs with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Reduce repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()