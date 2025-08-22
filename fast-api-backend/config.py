import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PDF_FOLDER = Path("pdfs")
STATE_FILE = Path("state/embeddings_state.pkl")
FAISS_INDEX_FILE = "state/faiss_index"

# Models
EMBEDDING_MODEL = "models/embedding-001"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Airtable
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
WEBHOOK_ID = os.getenv("WEBHOOK_ID")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")   
ENDORSEMENTS_TABLE_ID = os.getenv("AIRTABLE_ENDORSEMENTS_TABLE_ID")
ENDORSEMENT_TABLE_NAME = os.getenv("ENDORSEMENT_TABLE_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME =  os.getenv("PINECONE_INDEX_NAME")
