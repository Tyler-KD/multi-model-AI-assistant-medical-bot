import sys
import os
import json
from pathlib import Path

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import Config
from agents.rag_agent import MedicalRAG

# Custom JSON encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)

def test_table_queries():
    """Test the main application with table-specific queries."""
    # Get config
    config = Config()
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = MedicalRAG(
        config=config,
        llm=config.rag.llm,
        embedding_model=config.rag.embedding_model
    )
    
    # Define table-specific queries
    queries = [
        "What tables are in the document?",
        "Show me information from Table 1",
        "Tell me about the content in any tables in the document",
        "What data is presented in Table 6?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\n\nProcessing query: {query}")
        response = rag.process_query(query)
        
        # Check for tables in response
        if isinstance(response, dict):
            # Format and print response content
            if "response" in response:
                response_text = response["response"]
                if hasattr(response_text, "content"):
                    # It's an AIMessage
                    print(f"Response: {response_text.content}")
                else:
                    # It's a string or other format
                    print(f"Response: {response_text}")
            
            # Check sources
            if "sources" in response:
                table_sources = [s for s in response.get("sources", []) 
                               if isinstance(s, dict) and s.get("content_type") == "table"]
                
                print(f"\nTable sources in response: {len(table_sources)}")
                for i, source in enumerate(table_sources):
                    print(f"  {i+1}. {source}")
        else:
            print(f"Response (raw): {response}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    test_table_queries() 