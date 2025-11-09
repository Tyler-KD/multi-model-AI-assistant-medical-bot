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
        
        # Default response format instructions if not specified in config
        default_instructions = """
        When formatting your response:
        1. Use clear sections with headings when appropriate
        2. Present tabular data using proper markdown table formatting:
           | Header1 | Header2 | Header3 |
           |---------|---------|---------|
           | Data1   | Data2   | Data3   |
        3. For ordered lists, use numbered items
        4. For unordered lists, use bullet points
        5. Cite your sources when presenting specific information
        6. Use concise, professional medical language
        """
        
        # Get format instructions from config or use default
        self.response_format_instructions = getattr(config.rag, "response_format_instructions", default_instructions)
        self.include_sources = getattr(config.rag, "include_sources", True)
    
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
            # Format document - check if document has content directly or in payload format
            doc_text = doc.get("content", "").strip()
            
            # Handle metadata - might be directly in the document or as separate metadata field
            metadata = doc.get("metadata", {})
            if not metadata and isinstance(doc, dict):
                # If there's no metadata field but the document has keys like source, is_table, etc.
                # then the document itself contains the metadata fields directly
                metadata = doc
            
            source = metadata.get("source", "Unknown Source")
            
            # Check if this is a table
            is_table = metadata.get("is_table", False)
            content_type = metadata.get("content_type", "")
            table_index = metadata.get("table_index", None)
            
            # Format document based on its type
            if is_table or content_type == "table":
                # Ensure the table has proper markdown formatting
                clean_table_text = self._ensure_markdown_table_format(doc_text)
                table_name = f"Table {table_index}" if table_index is not None else "Table"
                
                # Preserve table formatting and add a special marker
                formatted_doc = (
                    f"[Document {i+1} - TABLE] The following information is presented as {table_name}:\n\n"
                    f"{clean_table_text}\n\n"
                    f"(Source: {source})"
                )
            else:
                # Standard document formatting
                formatted_doc = f"[Document {i+1}] {doc_text} (Source: {source})"
            
            # Check if adding this would exceed max context length
            if total_length + len(formatted_doc) > self.max_context_length:
                break
                
            context_parts.append(formatted_doc)
            total_length += len(formatted_doc)
        
        return "\n\n".join(context_parts)
    
    def _ensure_markdown_table_format(self, table_text: str) -> str:
        """
        Ensure the table text is properly formatted as a markdown table.
        
        Args:
            table_text: The original table text
            
        Returns:
            Properly formatted markdown table
        """
        # If already has pipe characters, it might already be in markdown format
        if "|" in table_text:
            lines = table_text.strip().split("\n")
            
            # Check if it has a header separator row
            has_separator = False
            for i, line in enumerate(lines):
                if i > 0 and re.match(r"^\s*\|[\s\-\|]+\|\s*$", line):
                    has_separator = True
                    break
            
            # If no separator row, try to add one after the first row
            if not has_separator and len(lines) > 1:
                # Get the number of columns by counting pipe characters in first row
                first_row = lines[0]
                col_count = first_row.count("|") - 1
                if col_count <= 0:
                    col_count = first_row.count("\t") + 1  # Try tab count
                
                # Create separator row
                separator = "|" + "|".join(["---"] * col_count) + "|"
                
                # Insert separator after header
                lines.insert(1, separator)
                
                # Ensure all rows start and end with pipe
                for i in range(len(lines)):
                    if not lines[i].startswith("|"):
                        lines[i] = "| " + lines[i]
                    if not lines[i].endswith("|"):
                        lines[i] = lines[i] + " |"
                
                return "\n".join(lines)
            
            # If already has pipe characters and separator, return as is
            return table_text
        
        # It's not in markdown format, try to convert it
        lines = table_text.strip().split("\n")
        if not lines:
            return table_text
        
        # Try to detect columns by splitting on whitespace
        rows = []
        for line in lines:
            if line.strip():
                # Try to split by tabs first, then by multiple spaces
                if "\t" in line:
                    rows.append(line.split("\t"))
                else:
                    rows.append(re.split(r"\s{2,}", line.strip()))
        
        if not rows:
            return table_text
        
        # Make all rows the same length
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")
        
        # Format as markdown table
        md_table_lines = []
        
        # Header row
        md_table_lines.append("| " + " | ".join(rows[0]) + " |")
        
        # Separator
        md_table_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        # Data rows
        for row in rows[1:]:
            md_table_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(md_table_lines)
    
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
        # Add table handling instructions if context contains table markers
        table_instructions = ""
        if "[Document" in context and "TABLE]" in context:
            table_instructions = """
            Some of the retrieved information is presented in table format. When using information from tables:
            1. Present tabular data using proper markdown table formatting with headers, like this:
               | Column1 | Column2 | Column3 |
               |---------|---------|---------|
               | Value1  | Value2  | Value3  |
            2. Preserve the original structure of tables when presenting data
            3. Clearly interpret the tabular data in your response
            4. Reference the relevant table when presenting specific data points
            5. If appropriate, summarize trends or patterns shown in the tables
            """
            
        # Build the prompt
        prompt = f"""You are a medical assistant providing accurate information based on verified medical sources.

        Here are the last few messages from our conversation:
        
        {chat_history}

        The user has asked the following question:
        {query}

        I've retrieved the following information to help answer this question:

        {context}

        {table_instructions}

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
            # Handle metadata - might be directly in the document or as separate metadata field
            metadata = doc.get("metadata", {})
            if not metadata and isinstance(doc, dict):
                # If there's no metadata field but the document has keys like source, is_table, etc.
                # then the document itself contains the metadata fields directly
                metadata = doc
                
            source = metadata.get("source", "Unknown Source")
            
            # Create a unique identifier for the source that includes if it's a table
            is_table = metadata.get("is_table", False)
            content_type = metadata.get("content_type", "")
            table_index = metadata.get("table_index", None)
            
            source_key = f"{source}_{is_table}_{table_index}" if (is_table or content_type == "table") else source
            
            # Skip duplicates
            if source_key in seen_sources:
                continue
                
            # Add source info
            source_info = {
                "title": source,
                "section": metadata.get("section", ""),
                "publication_date": metadata.get("publication_date", "")
            }
            
            # Add table-specific information if applicable
            if is_table or content_type == "table":
                table_descriptor = f"Table {table_index}" if table_index is not None else "Table"
                source_info["title"] = f"{source} ({table_descriptor})"
                source_info["content_type"] = "table"
            
            sources.append(source_info)
            seen_sources.add(source_key)
            
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
