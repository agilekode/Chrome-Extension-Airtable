import os
import requests
import tempfile
import pickle
from pathlib import Path
from fastapi import FastAPI,Request,HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
import faiss
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage

from fastapi.middleware.cors import CORSMiddleware
# Load environment
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
    )
# -------------------- CONFIG --------------------
PDF_FOLDER = Path("PDFS")

STATE_FILE = Path("embeddings_state.pkl")
EMBEDDING_MODEL = "models/embedding-001"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
WEBHOOK_ID = os.getenv("WEBHOOK_ID")
AIRTABLE_API_KEY =  os.getenv("AIRTABLE_API_KEY")   
ENDORSEMENTS_TABLE_ID = os.getenv("AIRTABLE_ENDORSEMENTS_TABLE_ID")
ENDORSEMENT_TABLE_NAME = os.getenv("ENDORSEMENT_TABLE_NAME")
cursor = 0
VECTOR_STORE = None
FAISS_INDEX_FILE = "faiss_index"


# ----------------------------Fast Api intializing ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
gemini_client = GoogleGenerativeAI(model="gemini-2.5-pro")  #



# -------------------- UTILS --------------------
def load_state():
    global VECTOR_STORE
    if Path(FAISS_INDEX_FILE).exists():
        VECTOR_STORE = FAISS.load_local(
            FAISS_INDEX_FILE,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[INFO] Loaded FAISS index")
    else:
        init_vector_store()

def save_state():
    if VECTOR_STORE:
        VECTOR_STORE.save_local(FAISS_INDEX_FILE)
        print("[INFO] Saved FAISS index")

def init_vector_store():
    """
    Initialize FAISS vector store (empty index).
    """
    global VECTOR_STORE

    # Get embedding dimension once by embedding a dummy string
    dim = len(embeddings.embed_query("hello world"))

    # Create an empty FAISS index
    index = faiss.IndexFlatL2(dim)

    VECTOR_STORE = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    print("[INFO] Initialized empty FAISS index.")

def ingest_pdfs():
    global VECTOR_STORE
    print("[INFO] Ingesting PDFs...")
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    all_texts, all_metas = [], []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        text = "\n".join(text_pages)
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metas.append({"filename": pdf_file.name, "chunk_index": i})

    # Build FAISS index
    VECTOR_STORE = FAISS.from_texts(all_texts, embedding=embeddings, metadatas=all_metas)
    save_state()
    print("[INFO] PDF ingestion complete.")


# -------------------- LOAD STATE ON START --------------------
load_state()
llm = get_llm()


# -------------------- API --------------------
class QueryRequest(BaseModel):
    text: str
    top_k: int = 3
    metadata : dict


# cosine similarity helper
def cosine_sim(a, b):
    from math import sqrt
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sqrt(sum(x*x for x in a))
    norm_b = sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b + 1e-8)
def search_by_metadata(metadata: dict, query: str, k: int = 10):
    if VECTOR_STORE is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    # Collect all filenames from metadata
    filenames = list(metadata.get("metadata",{}).values())
    print("Filenames are ",filenames)

    results = []
    for fname in filenames:
        docs = VECTOR_STORE.similarity_search(
            query=query,
            k=k,
            filter={"filename": fname}  # filter per filename
        )
        results.extend(docs)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in results
    ]

@app.post("/ask")
async def ask(request: QueryRequest):
    query_text = request.text
    metadata = request.metadata

    # 🔍 FAISS retrieval
    top_docs = search_by_metadata(metadata, query_text, k=10)

    # Build context from results
    context_text = ""
    for doc in top_docs:
        meta = doc["metadata"]
        context_text += f"[File: {meta.get('filename')}\n"
        context_text += doc["content"] + "\n\n"

    system_prompt = f"""
You are an expert assistant. Use the following PDF contents to answer the user query.
{context_text}
User Question: {request.text}
Answer with sources and references.
"""

    result = llm.invoke([HumanMessage(content=system_prompt)])
    gemini_answer = result.content

    return {
        "query": request.text,
        "gemini_answer": gemini_answer
    }





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


def pdf_already_embedded(pdf_filename: str) -> bool:
    """
    Check FAISS vector store for existing embeddings by filename (metadata lookup).
    """
    if VECTOR_STORE is None:
        return False

    # Iterate through stored docs metadata
    for doc_id, doc in VECTOR_STORE.docstore._dict.items():
        if doc.metadata.get("filename") == pdf_filename:
            print(f"✅ Found existing embedding for: {pdf_filename}")
            return True

    print(f"❌ No embedding found for: {pdf_filename}")
    return False


def extract_text_from_pdf(pdf_url: str) -> str:
    """
    Downloads a PDF from Airtable (authorized with API key), extracts text using PyPDF2.
    Returns extracted text as a string.
    """
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    # 1. Download PDF
    resp = requests.get(pdf_url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to download PDF: {resp.status_code} {resp.text}")

    # 2. Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(resp.content)
        tmp_path = tmp_file.name

    # 3. Extract text with PyPDF2
    text = []
    try:
        reader = PdfReader(tmp_path)
        print(f"PDF has {reader.pages}")
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    finally:
        # 4. Cleanup
        import os
        os.remove(tmp_path)

    return "\n".join(text)


def process_pdf_attachment(attachment: dict):
    """
    Process a single PDF attachment field from Airtable.
    """
    pdf_url = attachment.get("url")
    pdf_filename = attachment.get("filename")

    if not pdf_url or not pdf_filename:
        return False

    if pdf_already_embedded(pdf_filename):
        print(f"✅ PDF already embedded: {pdf_filename}")
        return False

    print(f"📥 Downloading + embedding new PDF: {pdf_filename}")
    pdf_text = extract_text_from_pdf(pdf_url)
    print(f"Len of pdf {pdf_filename} {len(pdf_text)}")
    VECTOR_STORE.add_texts([pdf_text], metadatas=[{"filename": pdf_filename}])
    save_state()
    return True


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


@app.post("/airtable-webhook")
async def airtable_webhook(request: Request):
    global cursor

    # Webhook ping from Airtable
    ping = await request.json()
    print("Webhook ping received:", ping)

    url = f"https://api.airtable.com/v0/bases/{AIRTABLE_BASE_ID}/webhooks/{WEBHOOK_ID}/payloads"
    if cursor:
        url += f"?cursor={cursor}"

    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    resp = requests.get(url, headers=headers)
    data = resp.json()

    # Update cursor for next fetch
    cursor = data.get("cursor", cursor)
    payloads = data.get("payloads", [])
    if not payloads:
        print("⚠️ No payloads found.")
        return {"status": "no changes"}

    # ✅ Only process the *latest* payload
    latest_payload = payloads[-1]

    changed_tables = latest_payload.get("changedTablesById", {})
    for table_id, table_changes in changed_tables.items():
        for record_change in table_changes.get("changedRecordsById", {}).values():
            record_id = record_change.get("id")
            fields = record_change.get("current", {}).get("cellValuesByFieldId", {})

            print(f"Changed record in table {table_id}")
            if check_and_embed_endorsement(fields):
                print(f"✅ PDFs embedded for record {record_id}")
            else:
                print(f"ℹ️ No endorsement PDFs found for record {record_id}")

    return {"status": "ok"}
    

@app.get("/search_policy")
def search_policy(policy_name: str):
    """
    Search for a record in Airtable by Policy Name and build metadata with base policy + endorsements.
    """
    SEARCH_COLUMN = "Policy Name"
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

    formula = f"{{{SEARCH_COLUMN}}}='{policy_name}'"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {"filterByFormula": formula, "maxRecords": 1}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    records = response.json().get("records", [])
    if not records:
        return {"message": "No matching records found."}

    record = records[0]
    fields = record.get("fields", {})

    meta_data = {}

    # 1. Get base policy filename
    attachments = fields.get("policy attachments", [])
    if attachments:
        base_filename = attachments[0].get("filename")
        meta_data["base_policy_filename"] = base_filename

    # 2. Loop over endorsements
    endorsement_ids = fields.get("Endorsements", [])
    for idx, endorsement_id in enumerate(endorsement_ids, start=1):
        end_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{ENDORSEMENT_TABLE_NAME}/{endorsement_id}"
        end_resp = requests.get(end_url, headers=headers)
        if end_resp.status_code != 200:
            continue  # skip if error
        end_record = end_resp.json()
        end_fields = end_record.get("fields", {})

        endorsement_files = end_fields.get("Endorsement PDF", [])
        if endorsement_files:
            filename = endorsement_files[0].get("filename")
            meta_data[f"endorsement_{idx}"] = filename

    return {"policy_name":policy_name ,"metadata":meta_data , "status_code":200}