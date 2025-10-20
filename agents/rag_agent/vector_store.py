from typing import List, Dict, Any, Optional, Union
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

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
        
        # Initialize Qdrant client
        if config.rag.use_local:
            self.client = QdrantClient(
                # location=config.rag.local_path
                path=config.rag.local_path
            )
        else:
            self.client = QdrantClient(
                url=config.rag.url,
                api_key=config.rag.api_key,
            )
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.logger.info(f"Creating new collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedding_dim,
                        distance=self.distance_metric,
                    ),
                    optimizers_config=qdrant_models.OptimizersConfigDiff(
                        indexing_threshold=10000,  # Optimize for production
                    ),
                )
                self.logger.info(f"Collection {self.collection_name} created successfully")
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise
    
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
                points.append(
                    qdrant_models.PointStruct(
                        id=doc["id"],
                        vector=doc["embedding"],
                        payload={
                            "content": doc["content"],
                            "source": doc["metadata"].get("source", ""),
                            "specialty": doc["metadata"].get("specialty", ""),
                            "section": doc["metadata"].get("section", ""),
                            "publication_date": doc["metadata"].get("publication_date", ""),
                            "medical_entities": doc["metadata"].get("medical_entities", []),
                            "chunk_number": doc["metadata"].get("chunk_number", 0),
                            "total_chunks": doc["metadata"].get("total_chunks", 1),
                        }
                    )
                )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            self.logger.info(f"Successfully upserted {len(documents)} documents")
        except Exception as e:
            self.logger.error(f"Error upserting documents: {e}")
            raise
    
    def retrieve(self, query_vector: List[float], filters: Optional[Dict] = None, 
                 top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on vector similarity and optional filters.
        
        Args:
            query_vector: Embedded query vector
            filters: Optional metadata filters
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with their metadata and relevance scores
        """
        try:
            filter_obj = None
            if filters:
                filter_conditions = []
                
                # Process specialty filter
                if "specialty" in filters and filters["specialty"]:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="specialty",
                            match=qdrant_models.MatchValue(value=filters["specialty"])
                        )
                    )
                
                # Process date range filter
                if "date_after" in filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="publication_date",
                            range=qdrant_models.Range(
                                gt=filters["date_after"]
                            )
                        )
                    )
                
                # Process medical entities filter
                if "medical_entities" in filters and filters["medical_entities"]:
                    for entity in filters["medical_entities"]:
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="medical_entities",
                                match=qdrant_models.MatchAny(any=[entity])
                            )
                        )
                
                # Combine all filters
                if filter_conditions:
                    filter_obj = qdrant_models.Filter(
                        must=filter_conditions
                    )
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filter_obj,
                limit=top_k,
                with_payload=True,
                score_threshold=0.6  # Minimum relevance threshold
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "content": result.payload["content"],
                    "metadata": {
                        "source": result.payload.get("source", ""),
                        "specialty": result.payload.get("specialty", ""),
                        "section": result.payload.get("section", ""),
                        "publication_date": result.payload.get("publication_date", ""),
                        "medical_entities": result.payload.get("medical_entities", []),
                        "chunk_number": result.payload.get("chunk_number", 0),
                        "total_chunks": result.payload.get("total_chunks", 1),
                    },
                    "score": result.score
                })
            
            return formatted_results
        
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            raise
    
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
