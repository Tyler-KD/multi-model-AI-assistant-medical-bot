import logging
import time
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
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
    def __init__(self, config, llm, embedding_model = None):
        """
        Initialize the Medical RAG system.
        
        Args:
            config: Configuration object
            llm: Language model for response generation
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.llm = llm
        
        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {config.rag.embedding_model}")
        # self.embedding_model = SentenceTransformer(config.rag.embedding_model, use_auth_token=config.rag.huggingface_token)
        self.embedding_model = embedding_model
        
        # Initialize components
        self.retriever = QdrantRetriever(config)
        self.document_processor = MedicalDocumentProcessor(config, self.embedding_model)
        self.query_processor = QueryProcessor(config, self.embedding_model)
        self.reranker = Reranker(config)
        self.response_generator = ResponseGenerator(config, llm)
        self.data_ingestion = MedicalDataIngestion(config_path=getattr(config, 'data_ingestion_config_path', None))
        # self.evaluator = RAGEvaluator(config)
        
        self.logger.info("Medical RAG system initialized successfully")

    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        """
        start_time = time.time()

        self.logger.info(f"Processing query: {query}")
        
        try:
            # Process query
            query_vector, filters = self.query_processor.process_query(query)

            # print("####### PRINTED from rag_agent/__init__.py: query_vector:", query_vector)
            
            # Temporarily disable filters until your documents have proper metadata
            filters = {}  # Comment this line out once you have documents with proper metadata
            
            # Retrieve documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(query_vector, filters)
            retrieval_time = time.time() - retrieval_start
            
            # print("####### PRINTED from rag_agent/__init__.py: retrieved_docs:", retrieved_docs)

            # Debug output
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs[:3]):  # Log first 3 docs
                self.logger.info(f"Doc {i}: Score {doc['score']}, Content: {doc['content'][:100]}...")
            
            # Rest of your code remains the same
            if retrieved_docs:
                reranked_docs = self.reranker.rerank(query, retrieved_docs)
            else:
                reranked_docs = []
            
            response_start = time.time()
            response = self.response_generator.generate_response(query, reranked_docs, chat_history)
            response_time = time.time() - response_start
            
            response["processing_time"] = time.time() - start_time
            response["num_docs_retrieved"] = len(retrieved_docs)  # Add this for debugging
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question.",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            documents: List of dictionaries with 'content' and 'metadata' keys
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting {len(documents)} documents")
        
        try:
            # Process documents
            processed_docs = []
            for doc in documents:
                chunks = self.document_processor.process_document(doc["content"], doc["metadata"])
                processed_docs.extend(chunks)
            
            # Generate embeddings
            embedding_start = time.time()
            chunk_texts = [chunk["content"] for chunk in processed_docs]
            embeddings = self.embedding_model.embed_documents(chunk_texts)
            embedding_time = time.time() - embedding_start
            
            # Add embeddings to processed documents
            for i, chunk in enumerate(processed_docs):
                chunk["embedding"] = embeddings[i]#.tolist()
            
            # Store documents in vector database
            storage_start = time.time()
            insertion_result = self.retriever.upsert_documents(processed_docs)
            storage_time = time.time() - storage_start
            
            # Log metrics
            metrics = {
                "documents_ingested": len(documents),
                "chunks_created": len(processed_docs),
                "embedding_time": embedding_time,
                "storage_time": storage_time,
                "total_processing_time": time.time() - start_time
            }
            
            return {
                "success": True,
                "metrics": metrics,
                "insertion_details": insertion_result
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
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