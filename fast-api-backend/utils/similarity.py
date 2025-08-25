from fastapi import HTTPException
from typing import List, Dict, Any

class VectorSearch:
    def __init__(self, vector_store , embeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings

    def search_by_metadata(
        self, metadata: dict, query: str, k: int = 10, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata and filter by similarity threshold.

        Args:
            metadata: dict containing "metadata" -> {key: filename}
            query: user query
            k: number of results to fetch per filename
            threshold: minimum similarity score to include a result

        Returns:
            List of dicts with content and metadata
        """
        if self.vector_store is None:
            raise HTTPException(status_code=500, detail="Vector store not initialized")

        filenames = list(metadata.get("metadata", {}).values())
        print(f"Query of User: {query}")
        print("Filenames are", filenames)

        results = []
        # for fname in filenames:
        #     docs_with_scores = self.vector_store.similarity_search_with_score(
        #         query=query,
        #         k=k,
        #         filter={"filename": fname}
        #     )

        #     for doc, score in docs_with_scores:
        #         print(f"score of docs {score}")
        #         # if score >= threshold:   # keep only results above threshold
        #         results.append((doc, score))
        filters = {"$or": [{"filename": fname} for fname in filenames]}
        query_embedding = self.embeddings.embed_query(query)
        docs_with_scores = self.vector_store.similarity_search_by_vector_with_score(
            embedding=query_embedding,
            k=k,
            filter=filters
        )
        print(f"Similarity results after filtering: {len(docs_with_scores)}")
        for doc, score in docs_with_scores:
            print(f"Filename: {doc.metadata}, Score: {score}")

        return [
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in docs_with_scores
        ]