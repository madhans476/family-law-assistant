"""
Standalone test script for retriever.py and Milvus connection.

This will help you diagnose issues with:
1. Milvus connection
2. Collection existence
3. Embedding generation
4. Search functionality
"""

import sys
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import traceback

# Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "family_law_cases"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def test_milvus_connection():
    """Test 1: Check if Milvus is running and accessible."""
    print("\n" + "="*60)
    print("TEST 1: Milvus Connection")
    print("="*60)
    
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   - Is Milvus running? Check with: docker ps | grep milvus")
        print(f"   - Try starting Milvus: docker-compose up -d")
        print(f"   - Check Milvus logs: docker-compose logs milvus-standalone")
        return False


def test_collection_exists():
    """Test 2: Check if collection exists."""
    print("\n" + "="*60)
    print("TEST 2: Collection Existence")
    print("="*60)
    
    try:
        if utility.has_collection(COLLECTION_NAME):
            print(f"✅ Collection '{COLLECTION_NAME}' exists")
            return True
        else:
            print(f"❌ Collection '{COLLECTION_NAME}' does not exist")
            print(f"\n💡 Troubleshooting:")
            print(f"   - Run: python milvus_store.py")
            print(f"   - This will create and populate the collection")
            return False
    except Exception as e:
        print(f"❌ Error checking collection: {e}")
        traceback.print_exc()
        return False


def test_collection_info():
    """Test 3: Get collection information."""
    print("\n" + "="*60)
    print("TEST 3: Collection Information")
    print("="*60)
    
    try:
        collection = Collection(COLLECTION_NAME)
        
        # Get collection stats
        collection.load()
        num_entities = collection.num_entities
        
        print(f"✅ Collection loaded successfully")
        print(f"   📊 Total entities: {num_entities}")
        
        if num_entities == 0:
            print(f"\n⚠️  Warning: Collection is empty!")
            print(f"   💡 Run: python milvus_store.py")
            return False
        
        # Get schema info
        schema = collection.schema
        print(f"   📋 Fields:")
        for field in schema.fields:
            print(f"      - {field.name}: {field.dtype}")
        
        return True
    except Exception as e:
        print(f"❌ Error getting collection info: {e}")
        traceback.print_exc()
        return False


def test_embedding_model():
    """Test 4: Check if embedding model loads."""
    print("\n" + "="*60)
    print("TEST 4: Embedding Model")
    print("="*60)
    
    try:
        print(f"Loading model: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        print(f"✅ Model loaded successfully")
        
        # Test encoding
        test_text = "test query"
        embedding = model.encode([test_text])[0]
        print(f"   📐 Embedding dimension: {len(embedding)}")
        print(f"   🔢 Sample values: {embedding[:5]}")
        
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return None


def test_search(model):
    """Test 5: Perform a test search."""
    print("\n" + "="*60)
    print("TEST 5: Search Functionality")
    print("="*60)
    
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Test query
        test_query = "How to file for divorce?"
        print(f"Test query: '{test_query}'")
        
        # Generate embedding
        query_embedding = model.encode([test_query])[0].tolist()
        print(f"✅ Query embedding generated ({len(query_embedding)} dimensions)")
        
        # Search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=5,
            output_fields=["content", "title", "category", "url"]
        )
        
        print(f"✅ Search completed")
        print(f"\n📋 Top {len(results[0])} results:")
        
        for i, hit in enumerate(results[0], 1):
            print(f"\n   Result {i}:")
            print(f"   Score: {hit.score:.4f}")
            print(f"   Title: {hit.entity.get('title', 'N/A')}")
            print(f"   Category: {hit.entity.get('category', 'N/A')}")
            print(f"   Content preview: {hit.entity.get('content', '')[:150]}...")
        
        return True
    except Exception as e:
        print(f"❌ Search failed: {e}")
        traceback.print_exc()
        return False


def test_retriever_function():
    """Test 6: Test the actual retriever function."""
    print("\n" + "="*60)
    print("TEST 6: Retriever Function")
    print("="*60)
    
    try:
        from nodes.retriever import retrieve_documents
        from state import FamilyLawState
        
        # Create test state
        state = {
            "query": "I want to file for divorce",
            "messages": [],
            "conversation_id": "test_123",
            "retrieved_chunks": [],
            "sources": []
        }
        
        print(f"Calling retrieve_documents with query: '{state['query']}'")
        
        result = retrieve_documents(state)
        
        retrieved_chunks = result.get("retrieved_chunks", [])
        sources = result.get("sources", [])
        
        print(f"✅ Retriever function executed")
        print(f"   📄 Retrieved chunks: {len(retrieved_chunks)}")
        print(f"   📚 Sources: {len(sources)}")
        
        if retrieved_chunks:
            print(f"\n   First chunk:")
            first_chunk = retrieved_chunks[0]
            print(f"   - Score: {first_chunk.get('score', 0):.4f}")
            print(f"   - Title: {first_chunk.get('metadata', {}).get('title', 'N/A')}")
            print(f"   - Content: {first_chunk.get('content', '')[:200]}...")
        else:
            print(f"\n   ⚠️  No chunks retrieved!")
        
        return len(retrieved_chunks) > 0
    except Exception as e:
        print(f"❌ Retriever function failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🔧"*30)
    print("RETRIEVER & MILVUS DIAGNOSTIC TESTS")
    print("🔧"*30)
    
    # Track results
    results = {}
    
    # Test 1: Connection
    results["connection"] = test_milvus_connection()
    if not results["connection"]:
        print("\n❌ Cannot proceed without Milvus connection")
        return
    
    # Test 2: Collection exists
    results["collection_exists"] = test_collection_exists()
    if not results["collection_exists"]:
        print("\n❌ Cannot proceed without collection")
        return
    
    # Test 3: Collection info
    results["collection_info"] = test_collection_info()
    
    # Test 4: Embedding model
    model = test_embedding_model()
    results["embedding_model"] = model is not None
    if not model:
        print("\n❌ Cannot proceed without embedding model")
        return
    
    # Test 5: Search
    results["search"] = test_search(model)
    
    # Test 6: Retriever function
    results["retriever_function"] = test_retriever_function()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! Retriever is working correctly.")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()