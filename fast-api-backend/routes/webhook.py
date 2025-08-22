from config import AIRTABLE_BASE_ID,WEBHOOK_ID,AIRTABLE_API_KEY,ENDORSEMENTS_TABLE_ID
import requests
import tempfile
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import APIRouter, Request
from core.embeddings import VECTOR_STORE,store_manager
from core.pdf_handler import PDFHandler

cursor = 0 
def fetch_endorsement_pdf(endorsement_id: str) -> list[str]:
    """
    Given an endorsement record ID, fetch its PDF attachment URLs.
    Returns list of URLs, or [] if none exist.
    """
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{ENDORSEMENTS_TABLE_ID}/{endorsement_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"⚠️ Failed to fetch endorsement {endorsement_id}: {resp.text}")
        return []

    record = resp.json()
    fields = record.get("fields", {})
    print(f"Fields {fields}")
    pdfs = fields.get("Endorsement PDF", [])

    urls = [{"url":att["url"],"filename":att["filename"]} for att in pdfs if att.get("type") == "application/pdf"]
    print(f"Pdf urls {urls}")
    return urls


# def pdf_already_embedded(pdf_filename: str) -> bool:
#     """
#     Check FAISS vector store for existing embeddings by filename (metadata lookup).
#     """
#     if VECTOR_STORE is None:
#         return False

#     # Iterate through stored docs metadata
#     for doc_id, doc in VECTOR_STORE.docstore._dict.items():
#         if doc.metadata.get("filename") == pdf_filename:
#             print(f"✅ Found existing embedding for: {pdf_filename}")
#             return True

#     print(f"❌ No embedding found for: {pdf_filename}")
#     return False


def pdf_already_embedded(pdf_filename: str) -> bool:
    if VECTOR_STORE is None:
        return False
    results = VECTOR_STORE.similarity_search(
        query="", 
        k=1, 
        filter={"filename": pdf_filename}
    )
    if results:
        print(f"✅ Found existing embedding for: {pdf_filename}")
        return True
    print(f"❌ No embedding found for: {pdf_filename}")
    return False


def process_pdf_attachment(attachment: dict) -> bool:
    """
    Process a single PDF attachment from Airtable:
    - Downloads the PDF
    - Extracts text
    - Splits into chunks
    - Embeds chunks into vector store
    """

    try:
        pdf_url = attachment.get("url")
        pdf_filename = attachment.get("filename")

        if not pdf_url or not pdf_filename:
            print("⚠️ Missing PDF URL or filename, skipping...")
            return False

        # Avoid duplicate embedding
        if pdf_already_embedded(pdf_filename):
            print(f"✅ Already embedded: {pdf_filename}")
            return False

        print(f"📥 Downloading + embedding new PDF: {pdf_filename}")
        pdf_text = PDFHandler.extract_text_from_url(pdf_url)

        if not pdf_text or len(pdf_text.strip()) == 0:
            print(f"⚠️ No text extracted from: {pdf_filename}")
            return False

        print(f"📝 Extracted {len(pdf_text)} characters from {pdf_filename}")

        # Use global splitter (do not reinitialize)
        global splitter
        if "splitter" not in globals():
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", " ", ""]
            )

        # Chunk text
        chunks = splitter.split_text(pdf_text)
        print(f"✂️ Split into {len(chunks)} chunks")

        if not chunks:
            print(f"⚠️ No chunks generated for {pdf_filename}")
            return False

        # Embed chunks
        VECTOR_STORE.add_texts(
            chunks,
            metadatas=[{"filename": pdf_filename}] * len(chunks)
        )

        store_manager.save()
        print(f"✅ Successfully embedded {pdf_filename}")
        return True

    except Exception as e:
        print(f"❌ Error processing {pdf_filename}: {e}")
        return False


def check_and_embed_endorsement(fields: dict) -> bool:
    """
    Check endorsement PDFs and attachment PDFs, embed only if not already embedded.
    """
    embedded = False
    # 1. Direct attachment field (fldGBZf9LrdpLcGbg)
    attachments = fields.get("fldGBZf9LrdpLcGbg") or []
    for att in attachments:
        if process_pdf_attachment(att):
            embedded = True

    # 2. Linked endorsement PDFs
    linked_endorsements = fields.get("fldHrkSa7uA9It6XL") or []
    for endorsement in linked_endorsements:
        endorsement_id = endorsement.get("id")
        pdf_urls = fetch_endorsement_pdf(endorsement_id)
        for pdf_dict in pdf_urls:
            url = pdf_dict.get("url")
            if process_pdf_attachment(pdf_dict):
                print(f"📄 Embedding PDF for endorsement {endorsement_id}: {url}")
                embedded = True

    return embedded


router = APIRouter()
@router.post("/airtable-webhook")
async def airtable_webhook(request: Request):
    global cursor

    ping = await request.json()
    print("Webhook ping received:", ping)

    url = f"https://api.airtable.com/v0/bases/{AIRTABLE_BASE_ID}/webhooks/{WEBHOOK_ID}/payloads"
    if cursor:
        url += f"?cursor={cursor}"

    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    resp = requests.get(url, headers=headers)
    data = resp.json()

    payloads = data.get("payloads", [])
    if not payloads:
        print("⚠️ No payloads found.")
        return {"status": "no changes"}

    # ✅ Process ALL payloads, not just the latest
    for payload in payloads:
        changed_tables = payload.get("changedTablesById", {})
        for table_id, table_changes in changed_tables.items():
            for record_change in table_changes.get("changedRecordsById", {}).values():
                record_id = record_change.get("id")
                fields = record_change.get("current", {}).get("cellValuesByFieldId", {})
                print(f"Changed record in table {table_id}")

                if check_and_embed_endorsement(fields):
                    print(f"✅ PDFs embedded for record {record_id}")
                else:
                    print(f"ℹ️ No endorsement PDFs found for record {record_id}")

    # ✅ Update cursor to the latest AFTER processing
    cursor = data.get("cursor", cursor)

    return {"status": "ok"}