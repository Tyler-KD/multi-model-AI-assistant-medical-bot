import sys
import os
from pathlib import Path
import json
import logging

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import Config

# Create a singleton Qdrant client manager
class QdrantClientManager:
    _instance = None
    _client = None
    
    @classmethod
    def get_client(cls, config):
        from qdrant_client import QdrantClient
        
        if cls._client is None:
            # Initialize only once
            if config.rag.use_local:
                print(f"Connecting to local Qdrant at: {config.rag.local_path}")
                cls._client = QdrantClient(path=config.rag.local_path)
            else:
                print(f"Connecting to remote Qdrant at: {config.rag.url}")
                cls._client = QdrantClient(url=config.rag.url, api_key=config.rag.api_key)
                
            print("Initialized Qdrant client singleton")
        
        return cls._client
    
    @classmethod
    def close(cls):
        if cls._client is not None:
            # Add closing logic if needed
            cls._client = None
            print("Closed Qdrant client connection")

# Patch QdrantRetriever to use our singleton
def patch_qdrant_retriever():
    from agents.rag_agent.vector_store import QdrantRetriever
    
    # Save the original init method
    original_init = QdrantRetriever.__init__
    
    # Create patched init that uses our singleton
    def patched_init(self, config):
        """Patched initializer that uses singleton client"""
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        
        # Use the singleton client
        self.client = QdrantClientManager.get_client(config)
        
        # Ensure collection exists
        self._ensure_collection()
    
    # Apply the patch
    QdrantRetriever.__init__ = patched_init
    print("Patched QdrantRetriever to use singleton client")

# Custom JSON encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)

def run_table_query(query):
    """
    Run a specific query targeting table content.
    
    Args:
        query: The query string to send to the RAG system
    """
    # Import required components
    from agents.rag_agent import MedicalRAG
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = Config()
    
    # Patch QdrantRetriever to use singleton client
    patch_qdrant_retriever()
    
    # Initialize models
    llm = config.rag.llm
    embedding_model = config.rag.embedding_model
    
    print(f"Initializing RAG system...")
    rag = MedicalRAG(config, llm, embedding_model=embedding_model)
    
    # Process the query
    print(f"\nProcessing query: {query}")
    response = rag.process_query(query)
    
    # Handle different response types
    if hasattr(response, 'get'):
        # Dictionary-like response
        print(f"Retrieved {response.get('num_docs_retrieved', 0)} documents")
        
        # Check sources for tables
        if "sources" in response:
            table_sources = []
            for source in response.get("sources", []):
                if isinstance(source, dict) and "Table" in source.get("title", ""):
                    table_sources.append(source)
            
            print(f"Table sources in response: {len(table_sources)}")
            for source in table_sources:
                print(f"  - {source}")
        
        # Show response text
        response_text = response.get('response', '')
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        print(f"\nResponse:")
        print(response_text)
        
    elif hasattr(response, 'content'):
        # LangChain AIMessage
        print("\nResponse (AIMessage):")
        print(response.content)
    else:
        # Unknown response format
        print("\nResponse (unknown format):")
        print(response)

def main():
    """Run test queries for tables."""
    # Define table-specific queries to test
    test_queries = [
        "What tables are in the document?",
        "Show me Table 1", 
        "Show me Table 2",
        "Show me Table 3", 
        "Show me Table 4",
        "Show me Table 5",
        "Show me Table 6"
    ]
    
    try:
        # Run only one query to avoid locking issues
        run_table_query(test_queries[3])
        print("\nTo test more queries, run this script again with a different query.")
    finally:
        # Ensure connection is closed
        QdrantClientManager.close()

if __name__ == "__main__":
    main() 