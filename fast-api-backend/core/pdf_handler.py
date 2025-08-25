from PyPDF2 import PdfReader
import tempfile, os, requests
import pymupdf4llm
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

        try:
            # pymupdf4llm extracts directly to markdown text
            markdown_text = pymupdf4llm.to_markdown(tmp_path)
            return markdown_text
        finally:
            os.remove(tmp_path)
