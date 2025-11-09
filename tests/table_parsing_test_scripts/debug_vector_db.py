import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create a singleton Qdrant client manager
class QdrantClientManager:
    _client = None
    
    @classmethod
    def get_client(cls, config):
        from qdrant_client import QdrantClient
        
        if cls._client is None:
            # Initialize only once
            if hasattr(config.rag, 'use_local') and config.rag.use_local:
                logger.info(f"Connecting to local Qdrant at: {config.rag.local_path}")
                cls._client = QdrantClient(path=config.rag.local_path)
            else:
                logger.info(f"Connecting to remote Qdrant at: {config.rag.url}")
                cls._client = QdrantClient(url=config.rag.url, api_key=config.rag.api_key)
        
        return cls._client
    
    @classmethod
    def close(cls):
        if cls._client is not None:
            # Add closing logic if needed
            cls._client = None
            logger.info("Closed Qdrant client connection")

# Custom JSON encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)

def serialize_json(obj):
    """Serialize object to JSON string using custom encoder."""
    return json.dumps(obj, cls=CustomJSONEncoder, indent=2)

def inspect_vector_db(collection_name=None, limit=5):
    """Inspect metadata of documents in the vector database."""
    
    # Load configuration
    config = Config()
    collection_name = collection_name or config.rag.collection_name
    
    # Get singleton client
    client = QdrantClientManager.get_client(config)
    
    logger.info(f"Inspecting collection: {collection_name}")
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        logger.error(f"Collection {collection_name} does not exist!")
        logger.info(f"Available collections: {collection_names}")
        return
    
    # Get total number of documents
    try:
        collection_info = client.get_collection(collection_name)
        total_documents = collection_info.vectors_count
        logger.info(f"Total documents in collection: {total_documents}")
    except Exception as e:
        logger.warning(f"Could not get collection info: {e}")
    
    # Search for table-related content
    logger.info("Searching for table-related content...")
    
    # Create a dummy vector of the right size for search
    vector_size = config.rag.embedding_dim if hasattr(config.rag, 'embedding_dim') else 1536
    dummy_vector = [0.1] * vector_size
    
    # Look for tables by checking content (for older clients)
    table_docs = []
    
    # First try direct search for "Table" keyword
    try:
        # Be careful not to search with query_text for older clients
        results_with_table = client.search(
            collection_name=collection_name,
            query_vector=dummy_vector,
            limit=100,  # Check more documents
            with_payload=True
        )
        
        # Filter results after retrieval
        for point in results_with_table:
            payload = point.payload if hasattr(point, 'payload') else {}
            content = payload.get("content", "")
            
            # Check if this looks like a table
            if "Table" in content or content.count("|") > 3:
                # Check if marked as table
                is_table = payload.get("is_table", False)
                content_type = payload.get("content_type", "")
                
                table_docs.append({
                    "id": point.id,
                    "content_preview": content[:100] + "...",
                    "is_table": is_table,
                    "content_type": content_type
                })
    except Exception as e:
        logger.warning(f"Error searching for 'Table': {e}")
    
    # Check for pipe character (table formatting)
    try:
        # Be careful not to search with query_text for older clients
        results_with_pipes = client.search(
            collection_name=collection_name,
            query_vector=dummy_vector,
            limit=100,
            with_payload=True
        )
        
        # Filter results after retrieval
        for point in results_with_pipes:
            # Skip if already found
            if any(doc["id"] == point.id for doc in table_docs):
                continue
                
            payload = point.payload if hasattr(point, 'payload') else {}
            content = payload.get("content", "")
            
            # Check if this has multiple pipe characters
            if content.count("|") > 3:
                # Check if marked as table
                is_table = payload.get("is_table", False)
                content_type = payload.get("content_type", "")
                
                table_docs.append({
                    "id": point.id,
                    "content_preview": content[:100] + "...",
                    "is_table": is_table, 
                    "content_type": content_type
                })
    except Exception as e:
        logger.warning(f"Error searching for '|': {e}")
    
    # Report on found tables
    if table_docs:
        logger.info(f"Found {len(table_docs)} table-like documents.")
        for i, doc in enumerate(table_docs[:limit]):
            logger.info(f"\nTable document {i+1}:")
            logger.info(f"ID: {doc['id']}")
            logger.info(f"Content: {doc['content_preview']}")
            logger.info(f"is_table: {doc['is_table']}")
            logger.info(f"content_type: {doc['content_type']}")
        
        if len(table_docs) > limit:
            logger.info(f"And {len(table_docs) - limit} more table documents...")
    else:
        logger.warning("No table-like documents found in the collection!")
    
    # Get sample documents to examine their metadata
    logger.info("\nRetrieving sample documents to examine metadata...")
    try:
        sample_results = client.search(
            collection_name=collection_name,
            query_vector=dummy_vector,
            limit=limit,
            with_payload=True
        )
        
        for i, point in enumerate(sample_results):
            logger.info(f"\nSample document {i+1} (ID: {point.id}):")
            
            payload = point.payload if hasattr(point, 'payload') else {}
            content = payload.get("content", "")
            logger.info(f"Content preview: {content[:100]}...")
            
            # Check what metadata fields are available
            payload_keys = list(payload.keys())
            logger.info(f"Payload keys: {payload_keys}")
            
            # Check table-specific metadata
            is_table = payload.get("is_table", None)
            if is_table is not None:
                logger.info(f"is_table: {is_table}")
            
            content_type = payload.get("content_type", None)
            if content_type is not None:
                logger.info(f"content_type: {content_type}")
            
            table_index = payload.get("table_index", None)
            if table_index is not None:
                logger.info(f"table_index: {table_index}")
    except Exception as e:
        logger.error(f"Error retrieving sample documents: {e}")

def main():
    """Main function to parse arguments and run inspection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Qdrant vector database contents")
    parser.add_argument("--collection", help="Name of the collection to inspect")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to show")
    
    args = parser.parse_args()
    
    try:
        inspect_vector_db(args.collection, args.limit)
    finally:
        # Ensure client is closed
        QdrantClientManager.close()

if __name__ == "__main__":
    main() 