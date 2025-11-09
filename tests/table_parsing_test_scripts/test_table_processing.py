import sys
import os
import json
from pathlib import Path

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from agents.rag_agent.data_ingestion import MedicalDataIngestion
from agents.rag_agent import MedicalRAG
from config import Config

config = Config()

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

def test_pdf_table_extraction(pdf_path):
    """Test table extraction from a PDF file."""
    print("\n=== Testing PDF Table Extraction ===")
    
    # Initialize data ingestion
    data_ingestion = MedicalDataIngestion()
    
    # Process the PDF file
    result = data_ingestion.ingest_file(pdf_path)
    
    if not result["success"]:
        print(f"❌ Error processing PDF: {result.get('error', 'Unknown error')}")
        return None
    
    print(f"✅ Successfully processed PDF: {pdf_path}")
    
    # Check if tables were extracted
    document = result["document"]
    tables = document.get("tables", [])
    
    print(f"Found {len(tables)} tables in the document")
    
    # Print details of the first few tables
    for i, table in enumerate(tables[:2]):  # Show first 2 tables only
        print(f"\nTable {i+1}:")
        print(f"Text preview: {table['text'][:200]}...")
        if "raw_text" in table:
            print(f"Raw text preview: {table['raw_text'][:100]}...")
        if "metadata" in table:
            print(f"Metadata: {serialize_metadata(table['metadata'])}")
    
    return document

def test_table_embedding(config, llm, embedding_model, document):
    """Test embedding of tables using the full RAG pipeline."""
    print("\n=== Testing Table Embedding ===")
    
    # Initialize the RAG system
    rag = MedicalRAG(config, llm, embedding_model)
    
    # Ingest the document with tables
    result = rag.ingest_documents([document])
    
    if not result["success"]:
        print(f"❌ Error ingesting document: {result.get('error', 'Unknown error')}")
        return
    
    print(f"✅ Successfully ingested document with tables")
    print(f"Metrics: {serialize_metadata(result['metrics'])}")
    
    # Get collection stats to verify table ingestion
    stats = rag.get_collection_stats()
    if stats["success"]:
        print(f"Collection stats: {serialize_metadata(stats['stats'])}")
    
    return rag

def test_table_retrieval(rag, table_related_query):
    """Test table retrieval using a query that should match table content."""
    print("\n=== Testing Table Retrieval ===")
    
    # Process the query
    response = rag.process_query(table_related_query)
    
    print(f"Query: {table_related_query}")
    
    # Handle different response types
    if hasattr(response, 'get'):
        # Dictionary-like response
        print(f"Retrieved {response.get('num_docs_retrieved', 0)} documents")
        
        # Get response text
        response_text = response.get('response', '')
        if not isinstance(response_text, str):
            response_text = str(response_text)
        print(f"Response: {response_text[:500]}...")
        
        # Check if any of the sources are tables
        if "sources" in response:
            table_sources = [s for s in response.get("sources", []) if "Table" in s.get("title", "")]
            print(f"Found {len(table_sources)} table sources in the response")
            for i, source in enumerate(table_sources):
                print(f"Table source {i+1}: {serialize_metadata(source)}")
                
    elif hasattr(response, 'content'):
        # LangChain AIMessage or similar
        print(f"Retrieved documents (count unknown - AIMessage response)")
        print(f"Response content: {response.content[:500]}...")
        # LangChain messages might not have table source metadata accessible
        print("Note: Table source information not available in AIMessage format")
        
    else:
        # Fallback for any other response type
        print(f"Response (unknown format): {str(response)[:500]}...")
    
    return response

def main():
    """Main test function."""
    # Check if a PDF path was provided
    if len(sys.argv) < 2:
        print("Usage: python test_table_processing.py <path_to_pdf_with_tables>")
        return
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return
    
    # Extract tables from PDF
    document = test_pdf_table_extraction(pdf_path)
    if not document:
        return
    
    # You'd need to set up your actual config, LLM, and embedding model here
    # This is just a placeholder - replace with your actual implementation
    try:
        llm = config.rag.llm
        embedding_model = config.rag.embedding_model
        
        # Test table embedding
        rag = test_table_embedding(config, llm, embedding_model, document)
        if not rag:
            return
        
        # Test table retrieval with a query that should match table content
        # Customize this query based on the content of your test PDF
        table_query = "Show me the data in the tables"
        test_table_retrieval(rag, table_query)
        
    except ImportError:
        print("\nSkipping embedding and retrieval tests.")
        print("To run full tests, you need to provide your config, LLM, and embedding model implementations.")
        print("Modify the test script to import your actual setup functions.")

if __name__ == "__main__":
    main() 