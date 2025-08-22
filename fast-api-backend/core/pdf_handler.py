from PyPDF2 import PdfReader
import tempfile, os, requests
from config import AIRTABLE_API_KEY

class PDFHandler:
    @staticmethod
    def extract_text_from_url(url: str) -> str:
        headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        text = []
        try:
            reader = PdfReader(tmp_path)
            for page in reader.pages:
                if (t := page.extract_text()):
                    text.append(t)
        finally:
            os.remove(tmp_path)

        return "\n".join(text)
