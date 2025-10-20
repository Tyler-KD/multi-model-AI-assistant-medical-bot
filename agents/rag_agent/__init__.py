import logging
import time
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

from .vector_store import QdrantRetriever
from .document_processor import MedicalDocumentProcessor
from .query_processor import QueryProcessor
from .reranker import Reranker
from .response_generator import ResponseGenerator
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
        # self.evaluator = RAGEvaluator(config)
        
        self.logger.info("Medical RAG system initialized successfully")
    
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()

        self.logger.info(f"Processing query: {query}")
        
        try:
            # Process query
            query_vector, filters = self.query_processor.process_query(query)
            
            # Retrieve documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(query_vector, filters)
            retrieval_time = time.time() - retrieval_start
            
            # Log retrieval
            # self.evaluator.log_retrieval(query, retrieved_docs, retrieval_time, success=(len(retrieved_docs) > 0))
            
            # Rerank documents if any were retrieved
            if retrieved_docs:
                reranked_docs = self.reranker.rerank(query, retrieved_docs)
            else:
                reranked_docs = []
            
            # Generate response
            response_start = time.time()
            response = self.response_generator.generate_response(query, reranked_docs, chat_history)
            response_time = time.time() - response_start
            
            # Log response
            # self.evaluator.log_response(query, response, response_time)
            
            # Calculate quality metrics
            # quality_metrics = self.evaluator.evaluate_response_quality(query, reranked_docs, response)
            
            # Add metrics to response
            # response["metrics"] = quality_metrics
            response["processing_time"] = time.time() - start_time
            
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
