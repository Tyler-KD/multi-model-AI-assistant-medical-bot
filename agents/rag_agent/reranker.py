import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
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
    
    def rerank(self, query: str, documents: Union[List[Dict[str, Any]], List[str]], parsed_content_dir: str) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance using cross-encoder.
        
        Args:
            query: User query
            documents: Either a list of documents (dictionaries) or a list of strings
            
        Returns:
            Reranked list of documents with updated scores
        """
        try:
            if not documents:
                return []
            
            # Handle different document formats and ensure consistent structure
            if documents:
                # if the retrieved documents is just a list of strings, we add a default score
                if isinstance(documents[0], str):
                    # Convert simple strings to dictionaries
                    docs_list = []
                    for i, doc_text in enumerate(documents):
                        docs_list.append({
                            "id": i,
                            "content": doc_text,
                            "score": 1.0  # Default score
                        })
                    documents = docs_list
                # if the retrieved documents is a list of dictionaries, we use the original score
                elif isinstance(documents[0], dict):
                    # Ensure all required fields exist in dictionaries
                    for i, doc in enumerate(documents):
                        # Ensure ID exists
                        if "id" not in doc:
                            doc["id"] = i
                        # Ensure score exists
                        if "score" not in doc:
                            doc["score"] = 1.0
                        # Ensure content exists (unlikely to be missing but just in case)
                        if "content" not in doc:
                            if "text" in doc:  # Some implementations might use "text" instead
                                doc["content"] = doc["text"]
                            else:
                                doc["content"] = f"Document {i}"
            
            # Create query-document pairs for scoring
            pairs = [(query, doc["content"]) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Add scores to documents
            for i, score in enumerate(scores):
                documents[i]["rerank_score"] = float(score)  # Store the new score from reranking
                # If the original document didn't have a score, use the rerank score
                if "score" not in documents[i]:
                    documents[i]["score"] = 1.0
                # Combine (average) the original score and rerank score
                documents[i]["combined_score"] = (documents[i]["score"] + float(score)) / 2
            
            # Sort by combined score
            reranked_docs = sorted(documents, key=lambda x: x["combined_score"], reverse=True)
            
            # Limit to top_k if needed
            if self.top_k and len(reranked_docs) > self.top_k:
                reranked_docs = reranked_docs[:self.top_k]
            
            # Extract picture references
            picture_reference_paths = []
            for doc in reranked_docs:
                matches = re.finditer(r"picture_counter_(\d+)", doc["content"])
                for match in matches:
                    counter_value = int(match.group(1))
                    # Create picture path based on document source and counter
                    doc_basename = os.path.splitext(doc['source'])[0]  # Remove file extension
                    # picture_path = Path(os.path.abspath(parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")).as_uri()
                    picture_path = os.path.join("http://localhost:8000/", parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")
                    picture_reference_paths.append(picture_path)
            
            return reranked_docs, picture_reference_paths
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            # Fallback to original ranking if reranking fails
            self.logger.warning("Falling back to original ranking")
            return documents