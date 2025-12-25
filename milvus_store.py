import os
import json
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm

# Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "family_law_cases"
EMBEDDINGS_DIR = "./data/embeddings"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

def connect_milvus():
    """Connect to Milvus standalone."""
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

def create_collection():
    """Create Milvus collection with schema."""
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️  Collection '{COLLECTION_NAME}' already exists. Dropping it...")
        utility.drop_collection(COLLECTION_NAME)
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="parent_id", dtype=DataType.INT64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="query_text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100)
    ]
    
    schema = CollectionSchema(fields=fields, description="Family Law Cases RAG")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    # Create index
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ Collection '{COLLECTION_NAME}' created with index")
    
    return collection

def insert_embeddings(collection):
    """Insert all embeddings into Milvus."""
    embedding_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith("_embeddings.json")]
    
    if not embedding_files:
        print("❌ No embedding files found. Run embedding.py first.")
        return
    
    total_inserted = 0
    
    for filename in embedding_files:
        category = filename.replace("_embeddings.json", "")
        file_path = os.path.join(EMBEDDINGS_DIR, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        print(f"\n📤 Inserting {len(chunks)} chunks from {category}")
        
        # Prepare data
        chunk_ids = []
        contents = []
        embeddings = []
        parent_ids = []
        titles = []
        query_texts = []
        urls = []
        categories = []
        
        for chunk in tqdm(chunks, desc=f"Preparing {category}"):
            chunk_ids.append(chunk["id"])
            contents.append(chunk["content"][:65535])  # Truncate if needed
            embeddings.append(chunk["embedding"])
            parent_ids.append(chunk["metadata"]["parent_id"])
            titles.append(chunk["metadata"]["title"][:1000])
            query_texts.append(chunk["metadata"]["query-text"][:10000])
            urls.append(chunk["metadata"].get("url", "")[:1000])
            categories.append(category)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(chunk_ids), batch_size):
            batch_data = [
                chunk_ids[i:i+batch_size],
                contents[i:i+batch_size],
                embeddings[i:i+batch_size],
                parent_ids[i:i+batch_size],
                titles[i:i+batch_size],
                query_texts[i:i+batch_size],
                urls[i:i+batch_size],
                categories[i:i+batch_size]
            ]
            collection.insert(batch_data)
        
        total_inserted += len(chunks)
        print(f"✅ Inserted {len(chunks)} chunks from {category}")
    
    collection.flush()
    print(f"\n🎯 Total {total_inserted} chunks inserted into Milvus")

def load_collection():
    """Load collection into memory."""
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"✅ Collection '{COLLECTION_NAME}' loaded into memory")
    return collection

if __name__ == "__main__":
    connect_milvus()
    collection = create_collection()
    insert_embeddings(collection)
    load_collection()
    print("\n✨ Milvus setup complete!")