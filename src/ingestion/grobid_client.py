from pathlib import Path
import requests


class GrobidClient:

    def __init__(self, url="http://localhost:8070"):
        self.url = url

    def process_pdf(self, pdf_path, output_path):
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)

        endpoint = f"{self.url}/api/processFulltextDocument"

        with open(pdf_path, "rb") as f:

            files = {
                "input": (
                    pdf_path.name,
                    f,
                    "application/pdf"
                )
            }

            response = requests.post(
                endpoint,
                files=files,
                timeout=120
            )

        response.raise_for_status()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path.write_text(
            response.text,
            encoding="utf-8"
        )

        return output_path