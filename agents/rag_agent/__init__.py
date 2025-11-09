import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import os
from pathlib import Path
import json
import datetime
import uuid
# from sentence_transformers import SentenceTransformer

from .vector_store import QdrantRetriever
from .document_processor import MedicalDocumentProcessor
from .query_processor import QueryProcessor
from .reranker import Reranker
from .response_generator import ResponseGenerator
from .data_ingestion import MedicalDataIngestion
# from .evaluation import RAGEvaluator

class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation system that integrates all components.
    """
    def __init__(self, config):
        """
        Initialize the RAG Agent.
        
        Args:
            config: Configuration object with RAG settings
        """
        self.config = config
        
        # Initialize components
        self._initialize()


    def _initialize(self):
        """Initialize the RAG components and load documents."""
        try:
            # Set up logging
            self.logger = logging.getLogger(f"{self.__module__}")
            self.logger.info("Initializing Medical RAG system")
            
            # Initialize LLM
            self.llm = self.config.rag.llm
            self.logger.info(f"Using LLM: {type(self.llm).__name__}")
            
            # Initialize embedding model
            self.embedding_model = self.config.rag.embedding_model
            self.logger.info(f"Using embedding model: {type(self.embedding_model).__name__}")
            
            # Initialize query processor
            self.query_processor = QueryProcessor(self.config, self.embedding_model)
            
            # Initialize document processor
            self.document_processor = MedicalDocumentProcessor(self.config, self.embedding_model)
            
            # Initialize vector store - using the QdrantRetriever class for consistency
            self.retriever = QdrantRetriever(self.config)
            
            # Initialize response generator with LLM
            self.response_generator = ResponseGenerator(self.config, self.llm)
            
            # Initialize reranker if configured
            self.reranker = None
            if hasattr(self.config.rag, "use_reranker") and self.config.rag.use_reranker:
                self.reranker = Reranker(self.config)
                self.logger.info("Using reranker for result refinement")
            
            # Verify vector store has documents
            total_docs = self.retriever.count_documents()
            self.logger.info(f"Vector store contains {total_docs} documents")
            if total_docs == 0:
                self.logger.warning("No documents in vector store. Results may be limited.")
                
            # Set default parameters
            self.top_k = getattr(self.config.rag, "top_k", 5)  # Default to 5 if not specified
            self.similarity_threshold = getattr(self.config.rag, "similarity_threshold", 0.0)
            
            # Initialize data ingestion component
            self.data_ingestion = MedicalDataIngestion()
            
            # Log initialization success
            self.logger.info("Medical RAG system successfully initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG system: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a query with the RAG system.
        
        Args:
            query: The query string
            chat_history: Optional chat history for context
            
        Returns:
            Response dictionary
        """
        self.logger.info(f"RAG Agent processing query: {query}")
        
        # Process query and return result, passing chat_history
        result = self.query(query, chat_history)
        
        return result
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting {len(documents)} documents")
        
        try:
            # Process each document using the document processor
            processed_documents = []
            document_ids = []
            
            for document in documents:
                content = document.get("content", "")
                metadata = document.get("metadata", {})
                
                # Add a unique ID if not present
                if "id" not in metadata:
                    metadata["id"] = str(uuid.uuid4())
                
                # Process the document
                processed_chunks = self.document_processor.process_document(content, metadata)
                
                if processed_chunks:
                    processed_documents.extend(processed_chunks)
                    for chunk in processed_chunks:
                        document_ids.append(chunk["id"])
                        
                        # Save processed document to the processed folder
                        processed_dir = Path("data/processed")
                        processed_dir.mkdir(exist_ok=True, parents=True)
                        
                        # Save as JSON file with document ID as name
                        doc_path = processed_dir / f"{chunk['id']}.json"
                        with open(doc_path, 'w', encoding='utf-8') as f:
                            # Ensure chunk is JSON serializable
                            json_safe_chunk = self._ensure_json_serializable(chunk)
                            json.dump(json_safe_chunk, f, indent=2)
            
            if not processed_documents:
                return {
                    "success": False,
                    "error": "No documents were successfully processed",
                    "processing_time": time.time() - start_time
                }
            
            # Prepare document embedding batch
            document_batch = []
            for doc in processed_documents:
                # Embed the content
                document_embedding = self.embedding_model.embed_documents([doc["content"]])[0]
                
                # Create document record for vector store
                document_record = {
                    "id": doc["id"],
                    "content": doc["content"],
                    "embedding": document_embedding,
                    "metadata": doc["metadata"]
                }
                
                document_batch.append(document_record)
            
            # Upsert into vector store
            self.retriever.upsert_documents(document_batch)
            
            return {
                "success": True,
                "documents_ingested": len(documents),
                "chunks_processed": len(processed_documents),
                "document_ids": document_ids,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting documents: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0,
                "chunks_created": 0,
                "chunks_inserted": 0,
                "processing_time": time.time() - start_time
            }
    
    def _ensure_json_serializable(self, obj):
        """
        Recursively convert objects to JSON-serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert non-serializable objects to string
            return str(obj)
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single file into the RAG system.
        
        Args:
            file_path: Path to the file to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting file: {file_path}")
        
        try:
            # Use the data ingestion component to process the file
            ingestion_result = self.data_ingestion.ingest_file(file_path)
            
            if not ingestion_result["success"]:
                return {
                    "success": False,
                    "error": ingestion_result.get("error", "Unknown error during file ingestion"),
                    "processing_time": time.time() - start_time
                }
            
            # Prepare documents for ingestion
            documents = []
            if "document" in ingestion_result:
                documents = [ingestion_result["document"]]
            elif "documents" in ingestion_result:
                documents = ingestion_result["documents"]
            
            # Ingest the documents
            if documents:
                return self.ingest_documents(documents)
            else:
                return {
                    "success": False,
                    "error": "No valid documents found in file",
                    "processing_time": time.time() - start_time
                }
            
        except Exception as e:
            self.logger.error(f"Error ingesting file: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def ingest_directory(self, directory_path: str, file_extension: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest all files in a directory into the RAG system.
        
        Args:
            directory_path: Path to the directory containing files
            file_extension: Optional file extension filter (e.g., ".txt", ".pdf")
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting directory: {directory_path}")
        
        try:
            # Use the data ingestion component to process all files in the directory
            directory_results = self.data_ingestion.ingest_directory(directory_path, file_extension)
            
            # Collect all documents from the ingestion results
            all_documents = []
            for file_path in Path(directory_path).glob(f"*{file_extension or ''}"):
                try:
                    ingestion_result = self.data_ingestion.ingest_file(str(file_path))
                    if ingestion_result["success"]:
                        if "document" in ingestion_result:
                            all_documents.append(ingestion_result["document"])
                        elif "documents" in ingestion_result:
                            all_documents.extend(ingestion_result["documents"])
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
            
            # Ingest all collected documents
            if all_documents:
                ingestion_result = self.ingest_documents(all_documents)
                ingestion_result["files_processed"] = directory_results["files_processed"]
                ingestion_result["errors"] = directory_results["errors"]
                return ingestion_result
            else:
                return {
                    "success": False,
                    "error": "No valid documents found in directory",
                    "files_processed": directory_results["files_processed"],
                    "errors": directory_results["errors"],
                    "processing_time": time.time() - start_time
                }
            
        except Exception as e:
            self.logger.error(f"Error ingesting directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def refresh_collection(self) -> Dict[str, Any]:
        """
        Refresh the vector database collection (e.g., optimize, update search index).
        
        Returns:
            Dictionary with refresh operation results
        """
        start_time = time.time()
        self.logger.info("Refreshing vector database collection")
        
        try:
            refresh_result = self.retriever.refresh_collection()
            
            return {
                "success": True,
                "details": refresh_result,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error refreshing collection: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current vector database collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = self.retriever.get_collection_stats()
            return {
                "success": True,
                "stats": stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def tune_retrieval_parameters(self, queries: List[str], expected_docs: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Tune the retrieval parameters based on a set of test queries and expected results.
        
        Args:
            queries: List of test queries
            expected_docs: List of lists of expected documents for each query
            
        Returns:
            Dictionary with tuning results and optimized parameters
        """
        start_time = time.time()
        self.logger.info(f"Tuning retrieval parameters with {len(queries)} test queries")
        
        try:
            # Initial evaluation
            initial_scores = []
            for i, query in enumerate(queries):
                query_vector, _ = self.query_processor.process_query(query)
                retrieved_docs = self.retriever.retrieve(query_vector)
                # score = self.evaluator.evaluate_retrieval(retrieved_docs, expected_docs[i])
                score = 0  # Placeholder for commented code
                initial_scores.append(score)
            
            initial_avg_score = sum(initial_scores) / len(initial_scores)
            
            # Parameter combinations to test
            param_combinations = [
                {"top_k": 5, "mmr_lambda": 0.7},
                {"top_k": 10, "mmr_lambda": 0.7},
                {"top_k": 5, "mmr_lambda": 0.5},
                {"top_k": 10, "mmr_lambda": 0.5}
            ]
            
            best_params = None
            best_score = initial_avg_score
            
            # Test each parameter combination
            for params in param_combinations:
                scores = []
                for i, query in enumerate(queries):
                    query_vector, _ = self.query_processor.process_query(query)
                    retrieved_docs = self.retriever.retrieve(
                        query_vector, 
                        top_k=params["top_k"],
                        mmr_lambda=params["mmr_lambda"]
                    )
                    # score = self.evaluator.evaluate_retrieval(retrieved_docs, expected_docs[i])
                    score = 0  # Placeholder for commented code
                    scores.append(score)
                
                avg_score = sum(scores) / len(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
            
            # Update config if better parameters were found
            if best_params and best_score > initial_avg_score:
                self.config.rag.retrieval.top_k = best_params["top_k"]
                self.config.rag.retrieval.mmr_lambda = best_params["mmr_lambda"]
                self.logger.info(f"Updated retrieval parameters: {best_params}")
            
            return {
                "success": True,
                "initial_score": initial_avg_score,
                "best_score": best_score,
                "best_parameters": best_params or "No improvement found",
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error tuning retrieval parameters: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all documents from the vector database collection.
        
        Returns:
            Dictionary with operation results
        """
        start_time = time.time()
        self.logger.info("Clearing vector database collection")
        
        try:
            clear_result = self.retriever.clear_collection()
            
            return {
                "success": True,
                "details": clear_result,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")

        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

    def process_ingested_data(self, process_type: str = "batch", **kwargs) -> Dict[str, Any]:
        """
        Process ingested data with various processing options.
        
        Args:
            process_type: Type of processing ('batch', 'incremental', 'priority')
            **kwargs: Additional parameters specific to the processing type
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        self.logger.info(f"Processing ingested data with method: {process_type}")
        
        try:
            if process_type == "batch":
                # Batch processing of all pending documents
                batch_size = kwargs.get("batch_size", 100)
                
                # In a real implementation, you would fetch pending documents from a queue
                # For now, we'll assume the documents are passed in kwargs
                documents = kwargs.get("documents", [])
                
                if not documents:
                    return {
                        "success": True,
                        "message": "No documents to process",
                        "processing_time": time.time() - start_time
                    }
                
                # Process in batches
                total_processed = 0
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    result = self.ingest_documents(batch)
                    if result["success"]:
                        total_processed += len(batch)
                
                return {
                    "success": True,
                    "documents_processed": total_processed,
                    "total_documents": len(documents),
                    "processing_time": time.time() - start_time
                }
                
            elif process_type == "incremental":
                # Process only new or modified documents
                last_update = kwargs.get("last_update", None)
                directory = kwargs.get("directory", None)
                
                if not directory:
                    return {
                        "success": False,
                        "error": "Directory required for incremental processing",
                        "processing_time": time.time() - start_time
                    }
                
                # Get modified files since last update
                new_or_modified_files = []
                if last_update:
                    for file_path in Path(directory).rglob("*"):
                        if file_path.is_file() and file_path.stat().st_mtime > last_update:
                            new_or_modified_files.append(str(file_path))
                else:
                    # If no last_update provided, treat all files as new
                    new_or_modified_files = [str(file_path) for file_path in Path(directory).rglob("*") if file_path.is_file()]
                
                # Process each file
                successful_files = 0
                for file_path in new_or_modified_files:
                    result = self.ingest_file(file_path)
                    if result["success"]:
                        successful_files += 1
                
                return {
                    "success": True,
                    "files_processed": successful_files,
                    "total_files": len(new_or_modified_files),
                    "processing_time": time.time() - start_time
                }
                
            elif process_type == "priority":
                # Process high-priority documents first
                priority_files = kwargs.get("priority_files", [])
                
                if not priority_files:
                    return {
                        "success": False,
                        "error": "No priority files specified",
                        "processing_time": time.time() - start_time
                    }
                
                # Process each priority file
                successful_files = 0
                for file_path in priority_files:
                    result = self.ingest_file(file_path)
                    if result["success"]:
                        successful_files += 1
                
                return {
                    "success": True,
                    "files_processed": successful_files,
                    "total_files": len(priority_files),
                    "processing_time": time.time() - start_time
                }
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown processing type: {process_type}",
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            self.logger.error(f"Error processing ingested data: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def analyze_source_quality(self, document_ids: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the quality of ingested sources based on various metrics.
        
        Args:
            document_ids: Optional list of document IDs to analyze
            
        Returns:
            Dictionary with quality analysis results
        """
        start_time = time.time()
        self.logger.info("Analyzing source quality")
        
        try:
            # Get statistics about the collection or specific documents
            if document_ids:
                # In a real implementation, fetch specific documents
                return {
                    "success": True,
                    "message": f"Quality analysis for {len(document_ids)} documents not implemented",
                    "processing_time": time.time() - start_time
                }
            else:
                # Get overall collection statistics
                stats = self.get_collection_stats()
                
                # In a real implementation, you would analyze these stats
                # For now, we'll return a placeholder
                
                return {
                    "success": True,
                    "collection_stats": stats.get("stats", {}),
                    "quality_metrics": {
                        "source_diversity": 0.85,  # Placeholder value
                        "content_freshness": 0.92,  # Placeholder value
                        "information_density": 0.78,  # Placeholder value
                        "content_specificity": 0.81   # Placeholder value
                    },
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing source quality: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def _retrieve_documents(self, query_embedding, filters, query):
        """
        Retrieve documents from vector store with enhanced filtering.
        
        Args:
            query_embedding: The query embedding vector
            filters: Metadata filters from query processing
            query: The original query text
            
        Returns:
            List of retrieved documents
        """
        # For debug: temporarily disable all filters while fixing the system
        # filters = {}
        
        self.logger.info(f"Retrieving documents with filters: {filters}")
        
        # Retrieve documents using vector search
        retrieved_docs = self.retriever.retrieve(
            query_vector=query_embedding, 
            filters=filters
        )
        
        self.logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    def query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a query and generate a response using RAG.
        
        Args:
            query: User query
            chat_history: Optional chat history for context
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        self.logger.info(f"Processing query: {query}")
        
        try:
            # Process the query
            query_embedding, filters = self.query_processor.process_query(query)
            
            # Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(query_embedding, filters, query)
            
            # Apply similarity threshold filtering
            if self.similarity_threshold > 0:
                retrieved_docs = [doc for doc in retrieved_docs if doc.get('score', 0) >= self.similarity_threshold]
                self.logger.info(f"After similarity threshold: {len(retrieved_docs)} documents")
            
            # Rerank if we have a reranker and enough documents
            if self.reranker and len(retrieved_docs) > 1:
                reranked_docs = self.reranker.rerank(query, retrieved_docs)
                self.logger.info(f"After reranking: {len(reranked_docs)} documents")
            else:
                reranked_docs = retrieved_docs
            
            # Generate response with the original method signature
            response = self.response_generator.generate_response(
                query=query, 
                retrieved_docs=reranked_docs,
                chat_history=chat_history
            )
            
            # Add timing information
            processing_time = time.time() - start_time
            response["processing_time"] = processing_time
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return error response
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }