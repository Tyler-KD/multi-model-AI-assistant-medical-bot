import sys
import os
import json
from pathlib import Path
import logging

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from agents.rag_agent.data_ingestion import MedicalDataIngestion
from agents.rag_agent.document_processor import MedicalDocumentProcessor
from agents.rag_agent import MedicalRAG

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("table_debug")

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

def debug_ingest_file(pdf_path):
    """Debug PDF ingestion to check if tables are extracted."""
    logger.info(f"Starting table extraction debug for: {pdf_path}")
    
    # Inject debug code into MedicalDataIngestion._ingest_pdf_file
    original_ingest_pdf = MedicalDataIngestion._ingest_pdf_file
    
    def debug_ingest_pdf_file(self, file_path):
        logger.info(f"[DEBUG] Calling _ingest_pdf_file for {file_path}")
        result = original_ingest_pdf(self, file_path)
        
        if result["success"]:
            doc = result["document"]
            tables = doc.get("tables", [])
            logger.info(f"[DEBUG] Extracted {len(tables)} tables from PDF")
            
            for i, table in enumerate(tables[:2]):  # Show only first 2 tables
                logger.info(f"[DEBUG] Table {i+1} preview: {table['text'][:100]}...")
                
                # Check table metadata - safely serialized
                if "metadata" in table:
                    logger.info(f"[DEBUG] Table {i+1} metadata: {serialize_metadata(table['metadata'])}")
        
        return result
    
    # Apply the debug patch
    MedicalDataIngestion._ingest_pdf_file = debug_ingest_pdf_file
    
    # Run the patched function
    data_ingestion = MedicalDataIngestion()
    result = data_ingestion.ingest_file(pdf_path)
    
    # Restore the original function
    MedicalDataIngestion._ingest_pdf_file = original_ingest_pdf
    
    return result

def debug_document_processing(document):
    """Debug document processing to check if tables are properly processed."""
    logger.info("Starting document processing debug")
    
    # We'll need a dummy embedding model for testing
    class DummyEmbeddingModel:
        def embed_documents(self, texts):
            logger.info(f"[DEBUG] Embedding {len(texts)} chunks")
            # Return dummy embeddings of the right dimension
            return [[0.01] * 1536 for _ in texts]
    
    # Create a dummy config
    class DummyConfig:
        class RAG:
            chunk_size = 1000
            chunk_overlap = 200
            chunking_strategy = "hybrid"
            processed_docs_dir = "./temp_processed_docs"
            
        rag = RAG()
    
    # Patch the document processor
    original_process_document = MedicalDocumentProcessor.process_document
    
    def debug_process_document(self, content, metadata):
        logger.info(f"[DEBUG] Processing document with metadata: {serialize_metadata(metadata)}")
        
        # Check if this is a table
        is_table = metadata.get("is_table", False)
        content_type = metadata.get("content_type", "")
        
        if is_table or content_type == "table":
            logger.info(f"[DEBUG] Processing TABLE content: {content[:100]}...")
        
        chunks = original_process_document(self, content, metadata)
        logger.info(f"[DEBUG] Created {len(chunks)} chunks")
        
        # Log details of the first chunk
        if chunks:
            first_chunk = chunks[0]
            logger.info(f"[DEBUG] First chunk preview: {first_chunk['content'][:100]}...")
            logger.info(f"[DEBUG] First chunk metadata: {serialize_metadata(first_chunk['metadata'])}")
        
        return chunks
    
    # Apply the debug patch
    MedicalDocumentProcessor.process_document = debug_process_document
    
    # Process the document
    dummy_config = DummyConfig()
    dummy_model = DummyEmbeddingModel()
    processor = MedicalDocumentProcessor(dummy_config, dummy_model)
    
    # First process the main content
    chunks = processor.process_document(document["content"], document["metadata"])
    logger.info(f"[DEBUG] Processed main content into {len(chunks)} chunks")
    
    # Then process tables if they exist
    if "tables" in document and document["tables"]:
        logger.info(f"[DEBUG] Processing {len(document['tables'])} tables")
        for i, table in enumerate(document["tables"]):
            # Create table-specific metadata
            table_metadata = document["metadata"].copy()
            table_metadata["content_type"] = "table"
            table_metadata["is_table"] = True
            table_metadata["table_index"] = i
            
            # Process the table
            table_chunks = processor.process_document(table["text"], table_metadata)
            logger.info(f"[DEBUG] Table {i+1} processed into {len(table_chunks)} chunks")
    
    # Restore the original function
    MedicalDocumentProcessor.process_document = original_process_document

def main():
    """Main debug function."""
    if len(sys.argv) < 2:
        print("Usage: python debug_table_processing.py <path_to_pdf_with_tables>")
        return
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return
    
    # Debug PDF ingestion
    result = debug_ingest_file(pdf_path)
    
    if not result["success"]:
        logger.error(f"Failed to ingest PDF: {result.get('error', 'Unknown error')}")
        return
    
    # Debug document processing
    document = result["document"]
    debug_document_processing(document)
    
    logger.info("Debugging complete. Check the logs above to verify table processing.")

if __name__ == "__main__":
    main() 