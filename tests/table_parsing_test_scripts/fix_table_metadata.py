import sys
import os
from pathlib import Path
import logging
import re

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_table_metadata(collection_name=None, dry_run=True):
    """
    Fix table entries in the vector database by adding the correct metadata flags.
    
    Args:
        collection_name: Name of the Qdrant collection to inspect
        dry_run: If True, only show what would be updated without making changes
    """
    try:
        # Import Qdrant client
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        from config import Config
        
        # Load configuration
        config = Config()
        
        # Connect to Qdrant
        if hasattr(config.rag, 'use_local') and config.rag.use_local:
            logger.info(f"Connecting to local Qdrant at: {config.rag.local_path}")
            client = QdrantClient(path=config.rag.local_path)
        else:
            logger.info(f"Connecting to remote Qdrant at: {config.rag.url}")
            client = QdrantClient(url=config.rag.url, api_key=config.rag.api_key)
        
        # Use collection name from config or provided parameter
        if collection_name is None:
            collection_name = config.rag.collection_name
        
        logger.info(f"Examining collection: {collection_name}")
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            logger.error(f"Collection {collection_name} does not exist")
            return
        
        # Create a dummy vector of the right size for search
        vector_size = config.rag.embedding_dim if hasattr(config.rag, 'embedding_dim') else 1536
        dummy_vector = [0.1] * vector_size
        
        # Search for potential table entries based on content pattern
        # Look for content containing "Table" or having markdown table formatting
        potential_table_points = []
        
        # Search for documents with "Table" in the content
        logger.info("Searching for documents containing 'Table'...")
        try:
            table_results = client.search(
                collection_name=collection_name,
                query_vector=dummy_vector,
                limit=100,
                with_payload=True,
                with_vectors=False,
                query_text="Table"
            )
            potential_table_points.extend(table_results)
            logger.info(f"Found {len(table_results)} documents containing 'Table'")
        except Exception as e:
            logger.warning(f"Error searching for 'Table': {e}")
        
        # Search for documents with "|" in the content (possible markdown tables)
        logger.info("Searching for documents containing '|' (possible markdown tables)...")
        try:
            pipe_results = client.search(
                collection_name=collection_name,
                query_vector=dummy_vector,
                limit=100,
                with_payload=True,
                with_vectors=False,
                query_text="|"
            )
            
            # Add only new points
            existing_ids = set(p.id for p in potential_table_points)
            for point in pipe_results:
                if point.id not in existing_ids:
                    potential_table_points.append(point)
                    existing_ids.add(point.id)
                    
            logger.info(f"Found {len(pipe_results)} documents containing '|'")
        except Exception as e:
            logger.warning(f"Error searching for '|': {e}")
        
        logger.info(f"Found total of {len(potential_table_points)} potential table entries based on content patterns")
        
        # Keep track of how many documents would be updated
        update_count = 0
        
        # List to store document updates
        updates = []
        
        # Check each potential table entry
        for point in potential_table_points:
            point_id = point.id
            payload = point.payload if hasattr(point, 'payload') else {}
            content = payload.get("content", "")
            
            # Skip if already marked as a table
            if payload.get("is_table") is True or payload.get("content_type") == "table":
                continue
            
            # Use more precise patterns to determine if this is actually a table
            is_table = False
            table_index = None
            
            # Pattern 1: Content contains markdown table formatting (multiple | characters)
            if content.count("|") > 3:
                is_table = True
                
            # Pattern 2: Content starts with "Table X" or contains "Table X."
            table_match = re.search(r"Table\s+(\d+)", content)
            if table_match:
                is_table = True
                table_index = int(table_match.group(1))
            
            if is_table:
                update_count += 1
                
                # Create payload with updated metadata
                payload_update = {
                    "is_table": True,
                    "content_type": "table"
                }
                
                if table_index is not None:
                    payload_update["table_index"] = table_index
                
                # Add to updates list
                updates.append((point_id, payload_update))
                
                # Log what we're updating
                logger.info(f"Document {point_id} will be marked as a table (index: {table_index})")
                logger.info(f"Content preview: {content[:100]}...")
        
        logger.info(f"\nFound {update_count} documents to update with table metadata")
        
        if not dry_run and updates:
            logger.info("Applying updates to the vector database...")
            
            # Process in batches to avoid overwhelming the database
            batch_size = 10
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]
                
                # Update each point individually to avoid API version issues
                for point_id, payload_update in batch:
                    try:
                        # Try different ways to update the payload based on client version
                        try:
                            # Method 1: Using set_payload with point_id directly
                            client.set_payload(
                                collection_name=collection_name,
                                points=[point_id],
                                payload=payload_update
                            )
                        except Exception as e1:
                            try:
                                # Method 2: Using update method
                                client.update(
                                    collection_name=collection_name,
                                    points=[(point_id, None, payload_update)]
                                )
                            except Exception as e2:
                                logger.error(f"Failed to update point {point_id}: {str(e1)} / {str(e2)}")
                        
                        logger.info(f"Updated point {point_id}")
                    except Exception as e:
                        logger.error(f"Error updating point {point_id}: {e}")
                
                logger.info(f"Updated batch {i//batch_size + 1}/{(len(updates)-1)//batch_size + 1}")
            
            logger.info(f"Successfully updated {update_count} documents with table metadata")
        elif dry_run:
            logger.info("Dry run completed. No changes were made to the database.")
            logger.info("Run with --apply to apply the changes.")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure qdrant_client is installed: pip install qdrant-client")
    except Exception as e:
        logger.error(f"Error fixing table metadata: {e}")

def main():
    """Main function to parse arguments and run the fix."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix table metadata in Qdrant vector database")
    parser.add_argument("--collection", type=str, default=None, help="Collection name to update")
    parser.add_argument("--apply", action="store_true", help="Actually apply the fixes (without this flag, runs in dry-run mode)")
    
    args = parser.parse_args()
    
    fix_table_metadata(args.collection, dry_run=not args.apply)

if __name__ == "__main__":
    main() 