import logging
from typing import List, Dict, Any, Optional
import re
import json
from collections import Counter

class RAGEvaluator:
    """
    Evaluates the performance of the RAG system and tracks metrics.
    """
    def __init__(self, config):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "queries_processed": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "avg_retrieval_time": 0,
            "avg_response_time": 0,
            "avg_confidence_score": 0,
            "feedback_scores": []
        }
        self.save_path = config.rag.metrics_save_path
    
    def log_retrieval(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                       retrieval_time: float, success: bool = True):
        """
        Log metrics for a retrieval operation.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            retrieval_time: Time taken for retrieval in seconds
            success: Whether retrieval was successful
        """
        self.metrics["queries_processed"] += 1
        
        if success and retrieved_docs:
            self.metrics["successful_retrievals"] += 1
            # Update average retrieval time
            prev_avg = self.metrics["avg_retrieval_time"]
            prev_count = self.metrics["queries_processed"] - 1
            self.metrics["avg_retrieval_time"] = (prev_avg * prev_count + retrieval_time) / self.metrics["queries_processed"]
            
            # Log confidence scores
            if retrieved_docs:
                scores = [doc.get("score", 0) for doc in retrieved_docs]
                avg_score = sum(scores) / len(scores) if scores else 0
                self.logger.info(f"Query: '{query}' | Docs: {len(retrieved_docs)} | Avg Score: {avg_score:.4f}")
        else:
            self.metrics["failed_retrievals"] += 1
            self.logger.warning(f"Failed retrieval for query: '{query}'")
    
    def log_response(self, query: str, response: Dict[str, Any], response_time: float):
        """
        Log metrics for a response generation operation.
        
        Args:
            query: User query
            response: Generated response
            response_time: Time taken for response generation in seconds
        """
        # Update average response time
        prev_avg = self.metrics["avg_response_time"]
        prev_count = self.metrics["queries_processed"] - 1
        self.metrics["avg_response_time"] = (prev_avg * prev_count + response_time) / self.metrics["queries_processed"]
        
        # Update average confidence score
        confidence = response.get("confidence", 0)
        prev_avg = self.metrics["avg_confidence_score"]
        self.metrics["avg_confidence_score"] = (prev_avg * prev_count + confidence) / self.metrics["queries_processed"]
        
        self.logger.info(f"Generated response for query: '{query}' | Confidence: {confidence:.4f}")
    
    def log_user_feedback(self, query: str, response: Dict[str, Any], feedback_score: int):
        """
        Log user feedback on responses.
        
        Args:
            query: User query
            response: Generated response
            feedback_score: User feedback score (1-5)
        """
        self.metrics["feedback_scores"].append({
            "query": query,
            "response": response.get("response", ""),
            "score": feedback_score
        })
        
        self.logger.info(f"Received feedback for query: '{query}' | Score: {feedback_score}")
    
    def evaluate_response_quality(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                                  response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of the response based on retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            response: Generated response
            
        Returns:
            Evaluation metrics
        """
        # Calculate metrics
        retrieval_precision = self._calculate_precision(query, retrieved_docs)
        answer_relevance = self._calculate_relevance(query, response, retrieved_docs)
        
        metrics = {
            "retrieval_precision": retrieval_precision,
            "answer_relevance": answer_relevance,
            "hallucination_risk": self._estimate_hallucination_risk(response, retrieved_docs),
            "answer_completeness": self._calculate_completeness(response, retrieved_docs)
        }
        
        return metrics
    
    def _calculate_precision(self, query: str, docs: List[Dict[str, Any]]) -> float:
        """
        Calculate precision of retrieved documents (simplified).
        
        Args:
            query: User query
            docs: Retrieved documents
            
        Returns:
            Precision score between 0 and 1
        """
        if not docs:
            return 0.0
        
        # Use scores as a proxy for relevance
        scores = [doc.get("score", 0) for doc in docs]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_relevance(self, query: str, response: Dict[str, Any], 
                             docs: List[Dict[str, Any]]) -> float:
        """
        Calculate relevance of the response to the query.
        
        Args:
            query: User query
            response: Generated response
            docs: Retrieved documents
            
        Returns:
            Relevance score between 0 and 1
        """
        if not docs or not response:
            return 0.0
        
        # Simple keyword-based relevance (in production, use more sophisticated methods)
        response_text = response.get("response", "").lower()
        query_words = set(query.lower().split())
        
        # Remove stopwords (simplified)
        stopwords = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        query_words = query_words - stopwords
        
        # Count query words in response
        word_count = 0
        for word in query_words:
            if word in response_text:
                word_count += 1
        
        return word_count / len(query_words) if query_words else 0.0
    
    def _estimate_hallucination_risk(self, response: Dict[str, Any], 
                                     docs: List[Dict[str, Any]]) -> float:
        """
        Estimate risk of hallucination in the response.
        
        Args:
            response: Generated response
            docs: Retrieved documents
            
        Returns:
            Hallucination risk score between 0 and 1 (higher is more risky)
        """
        if not docs or not response:
            return 1.0  # Maximum risk if no docs or response
        
        # Combine all document content
        all_doc_content = " ".join([doc["content"].lower() for doc in docs])
        response_text = response.get("response", "").lower()
        
        # Extract potential factual statements (sentences ending with period)
        factual_statements = re.findall(r'[^.!?]*[.!?]', response_text)
        
        # Count statements not supported by documents
        unsupported = 0
        total = len(factual_statements)
        
        for statement in factual_statements:
            # Simple check - statements with numbers or named entities are higher risk
            has_number = bool(re.search(r'\d+', statement))
            has_medical_term = bool(re.search(r'(?i)(disease|syndrome|treatment|medication|therapy|drug|dosage|diagnosis)', statement))
            
            if (has_number or has_medical_term) and not self._is_supported(statement, all_doc_content):
                unsupported += 1
        
        return unsupported / total if total > 0 else 0.5
    
    def _is_supported(self, statement: str, doc_content: str) -> bool:
        """
        Check if a statement is supported by document content.
        
        Args:
            statement: Statement to check
            doc_content: Document content
            
        Returns:
            True if supported, False otherwise
        """
        # Simple keyword matching (in production, use more sophisticated methods)
        keywords = statement.lower().split()
        keywords = [w for w in keywords if len(w) > 4]  # Only consider significant words
        
        if not keywords:
            return True
        
        # Check if at least 60% of keywords are found in doc content
        found = sum(1 for word in keywords if word in doc_content)
        return (found / len(keywords)) >= 0.6
    
    def _calculate_completeness(self, response: Dict[str, Any], 
                               docs: List[Dict[str, Any]]) -> float:
        """
        Calculate completeness of the response.
        
        Args:
            response: Generated response
            docs: Retrieved documents
            
        Returns:
            Completeness score between 0 and 1
        """
        response_text = response.get("response", "")
        
        # Heuristic based on response length and structure
        word_count = len(response_text.split())
        
        # Normalized score based on word count
        length_score = min(word_count / 150, 1.0)
        
        # Check for structural elements that suggest completeness
        has_introduction = bool(re.search(r'^[A-Z][^.!?]{10,}[.!?]', response_text))
        has_conclusion = bool(re.search(r'(?i)(in conclusion|to summarize|overall|in summary)', response_text))
        
        structure_score = (has_introduction + has_conclusion) / 2
        
        return (length_score * 0.7) + (structure_score * 0.3)
    
    def save_metrics(self):
        """Save current metrics to disk."""
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            self.logger.info(f"Metrics saved to {self.save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        retrieval_success_rate = 0
        if self.metrics["queries_processed"] > 0:
            retrieval_success_rate = self.metrics["successful_retrievals"] / self.metrics["queries_processed"]
        
        feedback_distribution = Counter(item["score"] for item in self.metrics["feedback_scores"])
        avg_feedback = sum(item["score"] for item in self.metrics["feedback_scores"]) / len(self.metrics["feedback_scores"]) if self.metrics["feedback_scores"] else 0
        
        return {
            "queries_processed": self.metrics["queries_processed"],
            "retrieval_success_rate": retrieval_success_rate,
            "avg_retrieval_time": self.metrics["avg_retrieval_time"],
            "avg_response_time": self.metrics["avg_response_time"],
            "avg_confidence_score": self.metrics["avg_confidence_score"],
            "feedback_distribution": feedback_distribution,
            "avg_feedback_score": avg_feedback
        }
