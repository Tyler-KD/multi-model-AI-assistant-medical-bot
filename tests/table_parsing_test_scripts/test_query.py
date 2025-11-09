# test_query.py
from agents.rag_agent import MedicalRAG
from config import Config

def main():
    # Load configuration
    config = Config()
    
    # Initialize RAG system with required models
    rag = MedicalRAG(
        config=config,
        llm=config.rag.llm,
        embedding_model=config.rag.embedding_model
    )
    
    # Process a query
    response = rag.process_query("What tables are in the document?")
    
    # Print response
    print("\nResponse:")
    print(response)
    
    # Check for table sources
    if isinstance(response, dict) and "sources" in response:
        table_sources = [s for s in response.get("sources", []) 
                         if isinstance(s, dict) and s.get("content_type") == "table"]
        print(f"\nTable sources found: {len(table_sources)}")
        for source in table_sources:
            print(f"  - {source}")

if __name__ == "__main__":
    main()