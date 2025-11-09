import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('table_processing_debug.log')
    ]
)

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

def add_debug_logging():
    """Add debug logging to key functions for table processing."""
    # Import necessary modules
    from agents.rag_agent.data_ingestion import MedicalDataIngestion
    from agents.rag_agent import MedicalRAG
    from agents.rag_agent.document_processor import MedicalDocumentProcessor
    from agents.rag_agent.response_generator import ResponseGenerator
    
    # Create logger
    logger = logging.getLogger("table_debug")
    logger.setLevel(logging.DEBUG)
    
    # 1. Patch _ingest_pdf_file to log table extraction
    original_ingest_pdf = MedicalDataIngestion._ingest_pdf_file
    
    def debug_ingest_pdf_file(self, file_path):
        logger.debug(f"[DEBUG] Ingesting PDF file: {file_path}")
        result = original_ingest_pdf(self, file_path)
        
        if result["success"]:
            doc = result["document"]
            tables = doc.get("tables", [])
            logger.debug(f"[DEBUG] Extracted {len(tables)} tables from PDF")
            
            for i, table in enumerate(tables):
                logger.debug(f"[DEBUG] Table {i+1} extracted, size: {len(table.get('text', ''))}")
                if "metadata" in table:
                    logger.debug(f"[DEBUG] Table {i+1} metadata sample: {serialize_metadata(table['metadata'])[:200]}...")
        
        return result
    
    # 2. Patch ingest_documents to log table processing
    original_ingest_documents = MedicalRAG.ingest_documents
    
    def debug_ingest_documents(self, documents):
        table_count = 0
        for doc in documents:
            if "tables" in doc and doc["tables"]:
                table_count += len(doc["tables"])
                
        logger.debug(f"[DEBUG] Ingesting {len(documents)} documents with {table_count} tables")
        result = original_ingest_documents(self, documents)
        
        if result["success"]:
            logger.debug(f"[DEBUG] Successfully processed {result['metrics'].get('chunks_created', 0)} chunks")
            
        return result
    
    # 3. Patch process_document to log table chunks
    original_process_document = MedicalDocumentProcessor.process_document
    
    def debug_process_document(self, content, metadata):
        is_table = metadata.get("is_table", False)
        content_type = metadata.get("content_type", "")
        source = metadata.get("source", "unknown")
        
        if is_table or content_type == "table":
            logger.debug(f"[DEBUG] Processing table content from {source}, size: {len(content)}")
            logger.debug(f"[DEBUG] Table metadata: {serialize_metadata(metadata)[:150]}...")
            
        chunks = original_process_document(self, content, metadata)
        
        if is_table or content_type == "table":
            logger.debug(f"[DEBUG] Table produced {len(chunks)} chunks")
            
        return chunks
    
    # 4. Patch _format_context to log when tables are included in context
    original_format_context = ResponseGenerator._format_context
    
    def debug_format_context(self, documents):
        table_count = 0
        for doc in documents:
            is_table = doc["metadata"].get("is_table", False)
            content_type = doc["metadata"].get("content_type", "")
            if is_table or content_type == "table":
                table_count += 1
                
        logger.debug(f"[DEBUG] Formatting context with {len(documents)} documents, including {table_count} tables")
        formatted_context = original_format_context(self, documents)
        
        return formatted_context
    
    # Apply all patches
    MedicalDataIngestion._ingest_pdf_file = debug_ingest_pdf_file
    MedicalRAG.ingest_documents = debug_ingest_documents
    MedicalDocumentProcessor.process_document = debug_process_document
    ResponseGenerator._format_context = debug_format_context
    
    print("✅ Debug logging has been added to table processing functions.")
    print("   Monitor 'table_processing_debug.log' for detailed information.")
    print("   To restore original functions, restart your application.")

if __name__ == "__main__":
    add_debug_logging() 