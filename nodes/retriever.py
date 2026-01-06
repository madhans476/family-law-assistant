from logging import root
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from typing import Dict
from state import FamilyLawState

# Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "family_law_cases"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Initialize model (loaded once)
model = SentenceTransformer(MODEL_NAME)

def connect_and_load():
    """Connect to Milvus and load collection."""
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection
    except Exception as e:
        print(f"❌ Error connecting to Milvus: {e}")
        return None

# Global collection instance
collection = connect_and_load()

def retrieve_documents(state: FamilyLawState) -> Dict:
    """
    Retrieve relevant documents from Milvus based on the query.
    """
    # Replace line 34 with this:
    root = state.get("root_query") or ""
    query = state.get("query") or ""
    combined_query = root + query

    query = combined_query

    if not collection:
        return {
            "retrieved_chunks": [],
            "sources": []
        }
    
    # Generate query embedding
    query_embedding = model.encode([query])[0].tolist()
    
    # Search in Milvus
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["content", "parent_id", "title", "query_text", "url", "category"]
    )
    
    # Process results
    retrieved_chunks = []
    sources = []
    
    for hits in results:
        for hit in hits:
            chunk_data = {
                "content": hit.entity.get("content"),
                "score": hit.score,
                "metadata": {
                    "parent_id": hit.entity.get("parent_id"),
                    "title": hit.entity.get("title"),
                    "query_text": hit.entity.get("query_text"),
                    "url": hit.entity.get("url"),
                    "category": hit.entity.get("category")
                }
            }
            retrieved_chunks.append(chunk_data)
            
            # Add unique sources
            source = {
                "title": hit.entity.get("title"),
                "url": hit.entity.get("url"),
                "category": hit.entity.get("category")
            }
            if source not in sources:
                sources.append(source)
    
    print(f"✅ Retrieved {len(retrieved_chunks)} chunks")
    
    return {
        "retrieved_chunks": retrieved_chunks,
        "sources": sources
    }