import logging
from typing import List, Dict, Any, Optional
import re

class ResponseGenerator:
    """
    Generates responses based on retrieved context and user query.
    """
    def __init__(self, config, llm):
        """
        Initialize the response generator.
        
        Args:
            config: Configuration object
            llm: Large language model for response generation
        """
        self.logger = logging.getLogger(__name__)
        self.llm = llm
        self.max_context_length = config.rag.max_context_length
        self.response_format_instructions = config.rag.response_format_instructions
        self.include_sources = config.rag.include_sources
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                          chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate a response based on retrieved documents and user query.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents
            chat_history: Optional chat history for continuity
            
        Returns:
            Dict containing response text and source information
        """
        try:
            if not retrieved_docs:
                return self._generate_no_documents_response(query)
            
            # print("####### PRINTED from rag_agent/response_generator.py: retrieved_docs:", retrieved_docs)
            
            # Format documents for context
            formatted_context = self._format_context(retrieved_docs)

            # print("####### PRINTED from rag_agent/response_generator.py: formatted_context:", formatted_context)
            
            # Build prompt
            prompt = self._build_prompt(query, formatted_context, chat_history)

            # print("####### PRINTED from rag_agent/response_generator.py: prompt:", prompt)
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Extract sources for citation
            sources = self._extract_sources(retrieved_docs) if self.include_sources else []
            
            # Format final response
            result = {
                "response": response,
                "sources": sources,
                "confidence": self._calculate_confidence(retrieved_docs)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0
            }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the language model.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents):
            # Format document
            doc_text = doc["content"].strip()
            source = doc["metadata"].get("source", "Unknown Source")
            
            # Add formatted document with source reference
            formatted_doc = f"[Document {i+1}] {doc_text} (Source: {source})"
            
            # Check if adding this would exceed max context length
            if total_length + len(formatted_doc) > self.max_context_length:
                # print("############### DEBUGGING ############: breaking out of formatting context loop due to low configured context length.")
                break
                
            context_parts.append(formatted_doc)
            total_length += len(formatted_doc)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, 
                      chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build the prompt for the language model.
        
        Args:
            query: User query
            context: Formatted context from retrieved documents
            chat_history: Optional chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_parts = []
            for exchange in chat_history[-3:]:  # Include last 3 exchanges at most
                if "user" in exchange and "assistant" in exchange:
                    history_parts.append(f"User: {exchange['user']}\nAssistant: {exchange['assistant']}")
            history_text = "\n\n".join(history_parts)
            history_text = f"Chat History:\n{history_text}\n\n"
            
        # Build the prompt
        prompt = f"""You are a medical assistant providing accurate information based on verified medical sources.
        
{history_text}The user has asked the following question:
"{query}"

I've retrieved the following information to help answer this question:

{context}

{self.response_format_instructions}

Based on the provided information, please answer the user's question thoroughly but concisely. If the information doesn't contain the answer, acknowledge the limitations of the available information.

Medical Assistant Response:"""

        return prompt
    
    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source = doc["metadata"].get("source", "Unknown Source")
            
            # Skip duplicates
            if source in seen_sources:
                continue
                
            # Add source info
            source_info = {
                "title": source,
                "section": doc["metadata"].get("section", ""),
                "publication_date": doc["metadata"].get("publication_date", "")
            }
            
            sources.append(source_info)
            seen_sources.add(source)
            
            # Limit to top 5 sources
            if len(sources) >= 5:
                break
                
        return sources
    
    def _generate_no_documents_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response when no relevant documents are found.
        
        Args:
            query: User query
            
        Returns:
            Response dictionary
        """
        prompt = f"""You are a medical assistant. The user has asked:
        "{query}"

        However, I don't have any specific information in my medical knowledge base to answer this question. 
        Please provide a general response acknowledging the limitations, and if appropriate, suggest what kind of medical professional they might consult.

        Medical Assistant Response:"""
        
        response = self.llm.invoke(prompt)
        
        return {
            "response": response,
            "sources": [],
            "confidence": 0.0
        }
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not documents:
            return 0.0
            
        # Use combined score if available, otherwise use original score
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:3]]
            
        # Average of top 3 document scores or fewer if less than 3
        return sum(scores) / len(scores) if scores else 0.0
