import os
from langchain_community.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone as PineconeClient ,ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from langchain_pinecone import PineconeVectorStore
from utils.similarity import VectorSearch
from langchain_openai import OpenAIEmbeddings


class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
        self.pinecone = PineconeClient(api_key=PINECONE_API_KEY)

    def load(self):
        # Check if index exists, else create it
        if PINECONE_INDEX_NAME not in self.pinecone.list_indexes().names():
            print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            # 768 is typical for "models/embedding-001", but better to confirm with one embedding
            dim = len(self.embeddings.embed_query("hello world"))
            self.pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                vector_type="dense",
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            deletion_protection="disabled",
            tags={
                "environment": "development"
            }
            )

        # Connect LangChain to Pinecone index
        index = self.pinecone.Index(PINECONE_INDEX_NAME)
        self.vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=self.embeddings
    )

    def save(self):
        # Pinecone auto-persists, no explicit save needed
        pass


# ---- Initialize once here ----
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",   # or "text-embedding-3-large"
    api_key=os.getenv("OPENAI_API_KEY")  # assumes OPENAI_API_KEY is set in env
)
store_manager = VectorStoreManager(embeddings)
store_manager.load()

VECTOR_STORE = store_manager.vector_store
vector_search = VectorSearch(VECTOR_STORE)
