import tempfile
import unittest
from pathlib import Path

from src.ingestion.chunker import create_chunks
from src.ingestion.grobid_parser import parse_grobid_tei


TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
 <teiHeader><fileDesc><titleStmt><title>Structured Evidence</title></titleStmt>
 <sourceDesc><biblStruct><analytic/></biblStruct></sourceDesc></fileDesc></teiHeader>
 <text><body><div><head>Results</head><p>The model was evaluated.</p>
  <figure type="table"><label>2</label><head>Main scores</head><table>
   <row><cell>Model</cell><cell>F1</cell></row><row><cell>Ours</cell><cell>91.2</cell></row>
  </table></figure>
  <figure><label>3</label><head>Accuracy by epoch</head><figDesc>The curve converges after ten epochs.</figDesc></figure>
  <formula xml:id="eq4"><label>4</label>E = mc²</formula>
 </div></body><back><div><listBibl><biblStruct xml:id="b1"><analytic><title>Prior Work</title></analytic><monogr><title>Journal</title></monogr></biblStruct></listBibl></div></back></text>
</TEI>"""


class StructuredElementTests(unittest.TestCase):
    def test_structured_elements_become_typed_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.tei.xml"
            path.write_text(TEI, encoding="utf-8")
            paper = parse_grobid_tei(path)

        chunks = create_chunks(paper, "paper.pdf")
        by_type = {chunk["element_type"]: chunk for chunk in chunks if chunk["element_type"] != "paragraph"}
        self.assertIn("Model | F1", by_type["table"]["text"])
        self.assertEqual(by_type["table"]["label"], "2")
        self.assertIn("converges after ten epochs", by_type["figure"]["text"])
        self.assertIn("E = mc²", by_type["equation"]["text"])
        self.assertIn("Prior Work", by_type["reference"]["text"])


if __name__ == "__main__":
    unittest.main()
