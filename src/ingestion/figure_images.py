"""Render GROBID-located figures from their source PDFs."""

import base64
from pathlib import Path

import pymupdf as fitz


def figure_data_url(pdf_path, coords, scale=2.0):
    """Crop ``page,x,y,width,height`` coordinates into a PNG data URL."""
    parts = [part.strip() for part in str(coords or "").split(",")]
    if len(parts) != 5:
        return None
    try:
        page_number = int(parts[0]) - 1
        x, y, width, height = map(float, parts[1:])
    except ValueError:
        return None
    path = Path(pdf_path)
    if not path.exists() or page_number < 0:
        return None
    with fitz.open(path) as document:
        if page_number >= len(document):
            return None
        page = document[page_number]
        clip = fitz.Rect(x, y, x + width, y + height) & page.rect
        if clip.is_empty:
            return None
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    return f"data:image/png;base64,{encoded}"
