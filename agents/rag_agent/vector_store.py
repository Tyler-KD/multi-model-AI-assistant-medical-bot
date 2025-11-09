from typing import List, Dict, Any, Optional, Union
import logging
import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Create a singleton Qdrant client manager
class QdrantClientManager:
    _instance = None
    _client = None
    
    @classmethod
    def get_client(cls, config):
        """Get or create a singleton Qdrant client instance."""
        if cls._client is None:
            # Initialize only once
            if config.rag.use_local:
                logging.info(f"Connecting to local Qdrant at: {config.rag.local_path}")
                cls._client = QdrantClient(path=config.rag.local_path)
            else:
                logging.info(f"Connecting to remote Qdrant at: {config.rag.url}")
                cls._client = QdrantClient(url=config.rag.url, api_key=config.rag.api_key)
                
            logging.info("Initialized Qdrant client singleton")
        
        return cls._client

class QdrantRetriever:
    """
    Handles storage and retrieval of medical documents using Qdrant vector database.
    """
    def __init__(self, config):
        """
        Initialize the Qdrant retriever with configuration.
        
        Args:
            config: Configuration object containing Qdrant settings
        """
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        
        # Use the singleton client instead of creating a new one
        self.client = QdrantClientManager.get_client(config)
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists, create it if it doesn't."""
        collection_info = self.client.get_collections()
        collection_names = [collection.name for collection in collection_info.collections]
        
        if self.collection_name not in collection_names:
            self.logger.info(f"Creating new collection {self.collection_name}")
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim, 
                        distance=Distance.COSINE
                    ),
                    optimizers_config=qdrant_models.OptimizersConfigDiff(
                        indexing_threshold=10000,  # Optimize for production
                    ),
                )
                self.logger.info(f"Collection {self.collection_name} created")
            except Exception as e:
                self.logger.error(f"Error creating collection: {e}")
                raise e
        else:
            self.logger.info(f"Collection {self.collection_name} already exists")
    
    def upsert_documents(self, documents: List[Dict[str, Any]]):
        """
        Insert or update documents in the vector database.
        
        Args:
            documents: List of document dictionaries containing:
                - id: Unique identifier
                - embedding: Vector embedding
                - metadata: Document metadata
                - content: Document content
        """
        try:
            points = []
            for doc in documents:
                # Extract all metadata for storage
                metadata = doc["metadata"].copy()
                
                # Create payload with standard fields and additional table-specific fields
                payload = {
                    "content": doc["content"],
                    "source": metadata.get("source", ""),
                    "specialty": metadata.get("specialty", ""),
                    "section": metadata.get("section", ""),
                    "publication_date": metadata.get("publication_date", ""),
                    "medical_entities": metadata.get("medical_entities", []),
                    "chunk_number": metadata.get("chunk_number", 0),
                    "total_chunks": metadata.get("total_chunks", 1),
                    # Add table-specific fields
                    "is_table": metadata.get("is_table", False),
                    "content_type": metadata.get("content_type", ""),
                    "table_index": metadata.get("table_index", None),
                }
                
                points.append(
                    qdrant_models.PointStruct(
                        id=doc["id"],
                        vector=doc["embedding"],
                        payload=payload
                    )
                )
            
            # Upsert in batches to prevent overloading
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            self.logger.info(f"Successfully upserted {len(documents)} documents")
        except Exception as e:
            self.logger.error(f"Error upserting documents: {e}")
            raise

    # def search_documents(self, query_embedding: List[float], top_k: int = 5, 
    #                     filter_condition: Optional[Dict] = None,
    #                     query_text: Optional[str] = None) -> List[Dict]:
    #     """
    #     Search for documents similar to the query embedding.
    #     Supports hybrid search (vector + text) when query_text is provided.
    #     Returns a list of metadata for the most similar documents.
    #     If filter_condition is provided, it will be used to filter the search results.
    #     """
    #     self.logger.info(f"Searching for documents with top_k={top_k}, filter={filter_condition}")
        
    #     filter_obj = None
    #     if filter_condition:
    #         # Process filter conditions
    #         conditions = []
    #         for key, value in filter_condition.items():
    #             # Convert to match values format for Qdrant
    #             if isinstance(value, list):
    #                 # Match any value in the list
    #                 conditions.append(
    #                     qdrant_models.FieldCondition(
    #                         key=key,
    #                         match=qdrant_models.MatchAny(any=value)
    #                     )
    #                 )
    #             else:
    #                 # Match exact value
    #                 conditions.append(
    #                     qdrant_models.FieldCondition(
    #                         key=key,
    #                         match=qdrant_models.MatchValue(value=value)
    #                     )
    #                 )
    #         # Combine all conditions (must satisfy all)
    #         filter_obj = qdrant_models.Filter(should=conditions)
        
    #     # Prepare search parameters
    #     search_params = {
    #         "collection_name": self.collection_name,
    #         "query_vector": query_embedding,
    #         "limit": top_k,
    #         "with_payload": True,
    #     }
        
    #     # Add filter if provided
    #     if filter_obj:
    #         search_params["filter"] = filter_obj
                
    #     # Implement hybrid search when query_text is provided
    #     if query_text:
    #         self.logger.info(f"Using hybrid search with text query: {query_text[:50]}...")
            
    #         # Configure text query - specify which fields to search in
    #         text_query = qdrant_models.TextQuery(
    #             text=query_text,
    #             fields=["content"]  # Specify fields to search in (adjust as needed)
    #         )
            
    #         # Configure hybrid search parameters
    #         hybrid_params = qdrant_models.SearchParams(
    #             hnsw_ef=128,  # Beam width for vector search part
    #             exact=False,  # Use approximate search for speed
    #             # Balance between vector and text scores (0.0-1.0)
    #             # 0.0 = vector only, 1.0 = text only, 0.7 = 70% vector, 30% text
    #             vector_ratio=0.7  
    #         )
            
    #         # Add hybrid search parameters to search request
    #         search_params["query_text"] = text_query
    #         search_params["search_params"] = hybrid_params
        
    #     self.logger.debug(f"Final search params: {search_params}")

    #     if filter_obj:
    #         self.logger.debug(f"Filter object details: {filter_obj.model_dump()}")
        
    #     try:
    #         search_results = self.client.search(**search_params)
            
    #         # Convert to standard format
    #         results = []
    #         for hit in search_results:
    #             # Combine payload and score
    #             doc = hit.payload.copy() if hit.payload else {}
    #             doc['score'] = hit.score
    #             results.append(doc)
            
    #         self.logger.info(f"Found {len(results)} matching documents")
    #         return results
    #     except Exception as e:
    #         self.logger.error(f"Error searching documents: {str(e)}")
    #         # Return empty list instead of raising to prevent system failure
    #         return []

    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self.logger.info(f"Collection {self.collection_name} deleted")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            raise e
    
    def count_documents(self) -> int:
        """Get the number of documents in the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            count = collection_info.vectors_count
            self.logger.info(f"Collection {self.collection_name} has {count} documents")
            return count
        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            return 0


    def retrieve(self, query_vector: List[float], filters: Optional[Dict] = None, 
                top_k: int = 5, include_metadata: bool = True, query_text: Optional[str] = None) -> List[Dict]:
        
        # Create filter object if filters provided
        filter_obj = None

        # TODO: Remove this once we have real filters
        filters = {}

        if filters and len(filters) > 0:
            try:
                conditions = []
                for key, value in filters.items():
                    # Add debug logging here:
                    self.logger.debug(f"Creating filter condition for key={key}, value={value}, type={type(value)}")
                    
                    # Handle nested dictionaries
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        for nested_key, nested_value in value.items():
                            # Try with dot notation
                            full_key = f"{key}.{nested_key}"
                            
                            # Log the exact key being used
                            self.logger.debug(f"Using nested key: {full_key} for value: {nested_value}")
                            
                            if isinstance(nested_value, list):
                                # For lists inside nested dictionaries
                                if len(nested_value) > 0:
                                    conditions.append(
                                        qdrant_models.FieldCondition(
                                            key=full_key,
                                            match=qdrant_models.MatchAny(any=nested_value)
                                        )
                                    )
                            else:
                                conditions.append(
                                    qdrant_models.FieldCondition(
                                        key=full_key,
                                        match=qdrant_models.MatchValue(value=nested_value)
                                    )
                                )
                    # Then continue with existing code for handling non-dict values
                    elif isinstance(value, list):
                        # Match any value in the list
                        if len(value) > 0:
                            conditions.append(
                                qdrant_models.FieldCondition(
                                    key=key,
                                    match=qdrant_models.MatchAny(any=value)
                                )
                            )
                    else:
                        # Match exact value
                        conditions.append(
                            qdrant_models.FieldCondition(
                                key=key,
                                match=qdrant_models.MatchValue(value=value)
                            )
                        )

                        # Add detailed debug logging after creating the condition:
                        self.logger.debug(f"Created MatchValue condition for key={key}")
                
                # Create the final filter object - all conditions must match
                filter_obj = qdrant_models.Filter(should=conditions)
                # Log the completed filter object
                self.logger.debug(f"Created filter with {len(conditions)} conditions")
                
            except Exception as e:
                self.logger.error(f"Error creating filter: {e}", exc_info=True)
                filter_obj = None
                
        # Search parameters
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }

        if filter_obj:
            search_params["query_filter"] = filter_obj
        
        # Add hybrid search parameters if query_text is provided
        if query_text:
            self.logger.info(f"Using hybrid search with text query: {query_text[:50]}...")
            
            # Configure text query - specify which fields to search in
            text_query = qdrant_models.TextQuery(
                text=query_text,
                fields=["content"]  # Specify fields to search in (adjust as needed)
            )
            
            # Configure hybrid search parameters
            hybrid_params = qdrant_models.SearchParams(
                hnsw_ef=128,
                exact=False,
                vector_ratio=0.7  # 70% vector, 30% text
            )
            
            # Add hybrid search parameters to search request
            search_params["query_text"] = text_query
            search_params["search_params"] = hybrid_params
            
        try:
            # Execute the search
            search_results = self.client.search(**search_params)
            
            # Convert to standard format
            results = []
            for hit in search_results:
                doc = hit.payload.copy() if hit.payload else {}
                doc['score'] = hit.score
                results.append(doc)
            
            self.logger.info(f"Found {len(results)} documents with hybrid search")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            # Return empty list instead of raising to prevent system failure
            return []


    def delete_documents(self, document_ids: List[Union[str, int]]):
        """
        Delete documents from the vector database by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=document_ids
                ),
                wait=True
            )
            self.logger.info(f"Successfully deleted {len(document_ids)} documents")
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            raise
            
    def wipe_collection(self):
        """Completely remove and recreate the collection for fresh start."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            self.logger.info(f"Collection {self.collection_name} wiped and recreated")
        except Exception as e:
            self.logger.error(f"Error wiping collection: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retrieve statistics of the collection.
        
        Returns:
            Dictionary containing collection statistics.
        """
        try:
            stats = self.client.get_collection(self.collection_name)
            self.logger.info(f"Collection stats retrieved successfully: {stats}")
            return stats.model_dump()
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            raise
