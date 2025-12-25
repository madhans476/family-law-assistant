import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Directories
CHUNKED_DIR = "./data/chunked"
EMBEDDINGS_DIR = "./data/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

def generate_embeddings(file_path):
    """Generate embeddings for all chunks in a file."""
    category_name = os.path.basename(file_path).replace("_chunks.json", "")
    output_file = os.path.join(EMBEDDINGS_DIR, f"{category_name}_embeddings.json")
    
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"\n📊 Processing {len(chunks)} chunks from {category_name}")
    
    # Extract content for batch processing
    contents = [chunk["content"] for chunk in chunks]
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    embeddings = model.encode(contents, show_progress_bar=True, batch_size=32)
    
    # Add embeddings to chunks
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        embedded_chunks.append({
            "id": i,
            "content": chunk["content"],
            "embedding": embeddings[i].tolist(),
            "metadata": chunk["metadata"]
        })
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(embedded_chunks, out, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(embedded_chunks)} embeddings → {output_file}")

if __name__ == "__main__":
    chunk_files = [f for f in os.listdir(CHUNKED_DIR) if f.endswith("_chunks.json")]
    
    if not chunk_files:
        print("❌ No chunk files found. Run chunking.py first.")
        exit(1)
    
    for filename in chunk_files:
        generate_embeddings(os.path.join(CHUNKED_DIR, filename))
    
    print("\n🎯 All embeddings generated and saved under:", EMBEDDINGS_DIR)