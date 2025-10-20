import warnings
warnings.filterwarnings('ignore')

import logging
import json
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent))

# Import your components
from agents.rag_agent import MedicalRAG
from config import Config

# Load configuration
config = Config()

# # Mock LLM for testing
# class MockLLM:
#     def generate(self, prompt, **kwargs):
#         # Simple mock response for testing
#         return {"text": f"This is a mock response for: {prompt[:50]}..."}

# # Initialize RAG system
# llm = MockLLM()
llm = config.rag.llm
embedding_model = config.rag.embedding_model
rag = MedicalRAG(config, llm, embedding_model = embedding_model)

# Test document ingestion
def test_ingestion():
    sample_docs = [
        {
            "content": "Diabetes mellitus is a disorder characterized by hyperglycemia...",
            "metadata": {"source": "medical_textbook", "topic": "diabetes", "specialty": "endocrinology"}
        },
        {
            "content": "Hypertension, also known as high blood pressure, is a long-term medical condition...",
            "metadata": {"source": "medical_journal", "topic": "hypertension", "specialty": "cardiology"}
        }
    ]
    
    result = rag.ingest_documents(sample_docs)
    print("Ingestion result:", json.dumps(result, indent=2))
    return result["success"]

# Test query processing
def test_query():
    queries = [
        "What are the symptoms of diabetes?",                           # related info has been ingested - output confidence medium/high
        "How is hypertension treated?",                                 # related info has been ingested - output confidence medium/high
        "What is the connection between diabetes and hypertension?"     # no direct info, but related topics exist - output confidence low
    ]
    
    for query in queries:
        print(f"\nTesting query: {query}")
        result = rag.process_query(query)
        print("Response:", result["response"])
        print("Sources:", len(result["sources"]))
        print("Confidence:", result["confidence"])
        print("Processing time:", result["processing_time"])
    
    return True

# Test collection stats
def test_stats():
    stats = rag.get_collection_stats()
    print("Collection stats:", json.dumps(stats, indent=2))
    return stats["success"]

# Run tests
if __name__ == "__main__":
    print("Starting RAG system tests...")
    
    print("\n1. Testing document ingestion...")
    ingestion_success = test_ingestion()
    
    if ingestion_success:
        print("\n2. Testing query processing...")
        query_success = test_query()
    
        if ingestion_success and query_success:
            print("\n3. Testing collection stats...")
            stats_success = test_stats()

            if ingestion_success and query_success and stats_success:
                print("\nAll tests completed.")
            
            else:
                print("Collection stats retrieval failed, skipping remaining tests.")
        
        else:
            print("Query processing failed, skipping remaining tests.")

    else:
        print("Ingestion failed, skipping remaining tests.")