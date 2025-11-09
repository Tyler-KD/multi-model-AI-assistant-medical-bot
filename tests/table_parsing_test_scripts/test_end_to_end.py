import sys
import os
import json
import logging
from pathlib import Path
import time

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("e2e_test")

# Custom JSON encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)

def serialize_metadata(metadata):
    """Safely serialize metadata by converting non-serializable objects."""
    if metadata is None:
        return "{}"
    
    try:
        return json.dumps(metadata, indent=2, cls=CustomJSONEncoder)
    except Exception as e:
        # Fallback for any other serialization issues
        return f"Metadata not serializable: {str(e)}"

def setup_environment():
    """Set up required environment for testing."""
    try:
        # Import your actual config and model setup
        # Modify these imports to match your project structure
        from config import Config
        
        # Load configuration
        config = Config()
        
        # Set up models
        llm = config.rag.llm
        embedding_model = config.rag.embedding_model
        
        return config, llm, embedding_model
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please modify the imports to match your project structure")
        return None, None, None

def test_table_e2e(pdf_path, config, llm, embedding_model):
    """
    End-to-end test of table processing, from ingestion to query.
    
    Args:
        pdf_path: Path to a PDF file with tables
        config: Config object
        llm: Language model
        embedding_model: Embedding model
    """
    from agents.rag_agent import MedicalRAG
    from agents.rag_agent.data_ingestion import MedicalDataIngestion
    
    # Step 1: Initialize RAG system
    logger.info("Initializing RAG system")
    rag = MedicalRAG(config, llm, embedding_model)
    
    # Step 2: Ingest the PDF file
    logger.info(f"Ingesting PDF file: {pdf_path}")
    start_time = time.time()
    result = rag.ingest_file(pdf_path)
    
    if not result["success"]:
        logger.error(f"Failed to ingest PDF: {result.get('error', 'Unknown error')}")
        return
    
    logger.info(f"✅ PDF ingestion successful in {time.time() - start_time:.2f} seconds")
    logger.info(f"Ingestion metrics: {serialize_metadata(result.get('metrics', {}))}")
    
    # Get collection stats
    stats = rag.get_collection_stats()
    if stats["success"]:
        logger.info(f"Collection stats: {serialize_metadata(stats['stats'])}")
    
    # Step 3: Verify table processing by querying for table content
    logger.info("\nTesting table-specific queries")
    
    # Prepare test queries
    test_queries = [
        "What tables are available in the document?",
        "Show me the data from the tables in the document",
        "Summarize the tabular data in the document"
    ]
    
    # Add document-specific queries if known
    # For example: "What is the value in the second row of the first table?"
    
    for query in test_queries:
        logger.info(f"\nProcessing query: {query}")
        start_time = time.time()
        response = rag.process_query(query)
        
        logger.info(f"Query processing time: {time.time() - start_time:.2f} seconds")
        
        # Handle different response types
        if hasattr(response, 'get'):
            # Dictionary-like response
            logger.info(f"Retrieved {response.get('num_docs_retrieved', 0)} documents")
            
            # Check if any of the sources are tables
            table_sources = []
            for source in response.get("sources", []):
                if "Table" in source.get("title", ""):
                    table_sources.append(source)
                    
            logger.info(f"Table sources in response: {len(table_sources)}")
            for source in table_sources:
                logger.info(f"  - {serialize_metadata(source)}")
            
            # Show response (truncated for readability)
            response_text = response.get('response', '')
            if not isinstance(response_text, str):
                # Handle case where response is an object
                response_text = str(response_text)
            logger.info(f"Response: {response_text[:300]}...")
            
        elif hasattr(response, 'content'):
            # LangChain AIMessage or similar
            logger.info(f"Retrieved documents (count unknown - AIMessage response)")
            logger.info(f"Response content: {response.content[:300]}...")
            
        else:
            # Fallback for any other response type
            logger.info(f"Response (unknown format): {str(response)[:300]}...")
    
    logger.info("\nEnd-to-end test completed successfully!")

def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python test_end_to_end.py <path_to_pdf_with_tables>")
        return
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return
    
    # Set up environment
    config, llm, embedding_model = setup_environment()
    if not (config and llm and embedding_model):
        print("Failed to set up environment. Please check the logs.")
        return
    
    # Run the end-to-end test
    test_table_e2e(pdf_path, config, llm, embedding_model)

if __name__ == "__main__":
    main() 