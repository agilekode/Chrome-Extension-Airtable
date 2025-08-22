from fastapi import HTTPException
from typing import List, Dict, Any

class VectorSearch:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search_by_metadata(self, metadata: dict, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            raise HTTPException(status_code=500, detail="Vector store not initialized")

        filenames = list(metadata.get("metadata", {}).values())
        print("Filenames are", filenames)

        results = []
        for fname in filenames:
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter={"filename": fname}
            )
            results.extend(docs)
        print(f"Similarity results {len(results)}")

        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]
