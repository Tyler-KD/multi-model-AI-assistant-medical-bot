import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Reranks retrieved documents using a cross-encoder model for more accurate results.
    """
    def __init__(self, config):
        """
        Initialize the reranker with configuration.
        
        Args:
            config: Configuration object containing reranker settings
        """
        self.logger = logging.getLogger(__name__)
        
        # Load the cross-encoder model for reranking
        # For medical data, specialized models like 'pritamdeka/S-PubMedBert-MS-MARCO'
        # would be ideal, but using a general one here for simplicity
        try:
            self.model_name = config.rag.reranker_model
            self.logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.top_k = config.rag.reranker_top_k
        except Exception as e:
            self.logger.error(f"Error loading reranker model: {e}")
            raise
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance using cross-encoder.
        
        Args:
            query: User query
            documents: List of documents from initial retrieval
            
        Returns:
            Reranked list of documents with updated scores
        """
        try:
            if not documents:
                return []
            
            # Create query-document pairs for scoring
            pairs = [(query, doc["content"]) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Add scores to documents
            for i, score in enumerate(scores):
                documents[i]["rerank_score"] = float(score)
                # Combine the original score and rerank score
                documents[i]["combined_score"] = (documents[i]["score"] + float(score)) / 2
            
            # Sort by combined score
            reranked_docs = sorted(documents, key=lambda x: x["combined_score"], reverse=True)
            
            # Limit to top_k if needed
            if self.top_k and len(reranked_docs) > self.top_k:
                reranked_docs = reranked_docs[:self.top_k]
            
            return reranked_docs
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            # Fallback to original ranking if reranking fails
            self.logger.warning("Falling back to original ranking")
            return documents
