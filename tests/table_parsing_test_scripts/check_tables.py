import sys
import os
import json
from pathlib import Path
import re

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import Config

# Create a singleton Qdrant client manager
class QdrantClientManager:
    _client = None
    
    @classmethod
    def get_client(cls, config):
        from qdrant_client import QdrantClient
        
        if cls._client is None:
            # Initialize only once
            if hasattr(config.rag, 'use_local') and config.rag.use_local:
                print(f"Connecting to local Qdrant at: {config.rag.local_path}")
                cls._client = QdrantClient(path=config.rag.local_path)
            else:
                print(f"Connecting to remote Qdrant at: {config.rag.url}")
                cls._client = QdrantClient(url=config.rag.url, api_key=config.rag.api_key)
        
        return cls._client
    
    @classmethod
    def close(cls):
        if cls._client is not None:
            # Add closing logic if needed
            cls._client = None
            print("Closed Qdrant client connection")

# Custom JSON encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)

def check_tables():
    """
    Check if tables have correct metadata values.
    Compatible with older Qdrant client versions.
    """
    # Load configuration
    config = Config()
    
    # Get singleton client
    client = QdrantClientManager.get_client(config)
    
    # Use collection name from config
    collection_name = config.rag.collection_name
    print(f"Checking collection: {collection_name}")
    
    # Create a dummy vector of the right size for search
    vector_size = config.rag.embedding_dim if hasattr(config.rag, 'embedding_dim') else 1536
    dummy_vector = [0.1] * vector_size
    
    # Use simple search without query_text (for older clients)
    print("Retrieving sample documents...")
    sample_results = client.search(
        collection_name=collection_name,
        query_vector=dummy_vector,
        limit=100,  # Retrieve more samples to find tables
        with_payload=True,
        with_vectors=False
    )
    
    # Analyze the results
    total_docs = len(sample_results)
    tables_found = 0
    table_flags_set = 0
    
    # Tables to update
    tables_needing_update = []
    
    print(f"Examining {total_docs} documents")
    
    for i, point in enumerate(sample_results):
        point_id = point.id
        payload = point.payload if hasattr(point, 'payload') else {}
        content = payload.get("content", "")
        
        # Check current metadata values
        is_table = payload.get("is_table", False)
        content_type = payload.get("content_type", "")
        
        # Check if content looks like a table
        looks_like_table = False
        table_index = None
        
        # Pattern 1: Content contains markdown table formatting (multiple | characters)
        if content.count("|") > 3:
            looks_like_table = True
            
        # Pattern 2: Content starts with "Table X" or contains "Table X."
        table_match = re.search(r"Table\s+(\d+)", content)
        if table_match:
            looks_like_table = True
            table_index = int(table_match.group(1))
        
        if looks_like_table:
            tables_found += 1
            
            # Check if metadata is already set correctly
            if is_table and (content_type == "table"):
                table_flags_set += 1
            else:
                # This table needs its metadata updated
                tables_needing_update.append({
                    "id": point_id,
                    "content_preview": content[:100] + "...",
                    "current_is_table": is_table,
                    "current_content_type": content_type,
                    "table_index": table_index
                })
    
    # Print results
    print(f"\nResults:")
    print(f"Total documents examined: {total_docs}")
    print(f"Tables found: {tables_found}")
    print(f"Tables with correct metadata: {table_flags_set}")
    print(f"Tables needing metadata update: {len(tables_needing_update)}")
    
    # Show tables needing updates
    if tables_needing_update:
        print("\nTables needing metadata updates:")
        for i, table in enumerate(tables_needing_update[:5]):  # Show first 5
            print(f"\nTable {i+1} (ID: {table['id']}):")
            print(f"Content: {table['content_preview']}")
            print(f"Current is_table: {table['current_is_table']}")
            print(f"Current content_type: {table['current_content_type']}")
            print(f"Detected table index: {table['table_index']}")
        
        if len(tables_needing_update) > 5:
            print(f"\n... and {len(tables_needing_update) - 5} more tables needing updates")
    
    return tables_needing_update

def update_table_metadata(tables_to_update):
    """
    Update table metadata for specific documents.
    
    Args:
        tables_to_update: List of dictionaries with table information
    """
    if not tables_to_update:
        print("No tables to update.")
        return
    
    # Load configuration
    config = Config()
    
    # Get singleton client
    client = QdrantClientManager.get_client(config)
    
    # Use collection name from config
    collection_name = config.rag.collection_name
    
    print(f"Updating {len(tables_to_update)} tables in collection: {collection_name}")
    
    # Update each table
    for i, table in enumerate(tables_to_update):
        point_id = table["id"]
        
        # Create payload with updated metadata
        payload_update = {
            "is_table": True,
            "content_type": "table"
        }
        
        if table["table_index"] is not None:
            payload_update["table_index"] = table["table_index"]
        
        try:
            # Try using set_payload method
            client.set_payload(
                collection_name=collection_name,
                points=[point_id],
                payload=payload_update
            )
            print(f"Updated table {i+1}/{len(tables_to_update)} (ID: {point_id})")
        except Exception as e:
            print(f"Error updating table {point_id}: {e}")
    
    print(f"Completed updating {len(tables_to_update)} tables")

def main():
    """Main function to run check and update if requested."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check and fix table metadata")
    parser.add_argument("--update", action="store_true", help="Update table metadata (without this flag, only checks)")
    
    args = parser.parse_args()
    
    try:
        # Check tables first
        tables_to_update = check_tables()
        
        # Update if requested
        if args.update and tables_to_update:
            print("\nUpdating table metadata...")
            update_table_metadata(tables_to_update)
    finally:
        # Ensure client is closed
        QdrantClientManager.close()

if __name__ == "__main__":
    main() 