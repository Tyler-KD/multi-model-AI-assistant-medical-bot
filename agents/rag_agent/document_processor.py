import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
import hashlib
from datetime import datetime

class MedicalDocumentProcessor:
    """
    Processes medical documents for the RAG system with context-aware chunking.
    """
    def __init__(self, config, embedding_model):
        """
        Initialize the document processor.
        
        Args:
            config: Configuration object
            embedding_model: Model to generate embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
        self.chunk_size = config.rag.chunk_size
        self.chunk_overlap = config.rag.chunk_overlap
        self.processed_docs_dir = Path(config.rag.processed_docs_dir)
        self.processed_docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Medical section headers pattern
        self.section_headers = [
            r"(?i)introduction",
            r"(?i)abstract",
            r"(?i)background",
            r"(?i)methods?",
            r"(?i)results?",
            r"(?i)discussion",
            r"(?i)conclusion",
            r"(?i)references",
            r"(?i)clinical presentation",
            r"(?i)diagnosis",
            r"(?i)treatment",
            r"(?i)prognosis",
            r"(?i)etiology",
            r"(?i)epidemiology",
            r"(?i)pathophysiology",
            r"(?i)signs and symptoms",
            r"(?i)complications",
            r"(?i)prevention",
            r"(?i)patient education"
        ]
        # self.section_pattern = re.compile(f"({'|'.join(self.section_headers)})")

        filtered_headers = [re.escape(header) for header in self.section_headers if header.strip()]
        self.section_pattern = re.compile(f"({'|'.join(filtered_headers)})")
        
        # Medical entities pattern (simplified - in production would use a medical NER model)
        self.medical_entity_pattern = re.compile(
            r"(?i)(diabetes|hypertension|cancer|asthma|covid-19|stroke|"
            r"alzheimer's|parkinson's|arthritis|obesity|heart disease|hepatitis|"
            r"influenza|pneumonia|tuberculosis|hiv/aids|malaria|cholera|"
            r"diabetes mellitus|chronic kidney disease|copd)"
        )
    
    def process_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a document and create chunks with embeddings.
        
        Args:
            content: Document content string
            metadata: Document metadata including source, specialty, etc.
            
        Returns:
            List of processed document chunks with embeddings
        """
        try:
            # Create document ID based on content hash
            doc_id_base = hashlib.md5(content.encode()).hexdigest()
            doc_id = str(uuid.UUID(doc_id_base[:32]))  # Convert first 32 chars to valid UUID
            
            # Extract medical entities
            medical_entities = self._extract_medical_entities(content)
            
            # Add entities to metadata
            enhanced_metadata = metadata.copy()
            enhanced_metadata['medical_entities'] = medical_entities
            
            # Create context-aware chunks
            chunks = self._create_medical_context_chunks(content)
            
            # Process each chunk
            processed_chunks = []
            for i, (chunk_text, section) in enumerate(chunks):
                # chunk_id = f"{doc_id_base}_{i}"
                # Generate chunk ID as a UUID with a suffix
                chunk_id = str(uuid.UUID(doc_id_base[:24] + f"{i:08}"))  # Append chunk index safely
                
                # Generate embedding
                embedding = self.embedding_model.embed_documents(chunk_text)
                
                # Create chunk metadata
                chunk_metadata = enhanced_metadata.copy()
                chunk_metadata["chunk_number"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["section"] = section
                
                # Create processed chunk
                processed_chunks.append({
                    "id": chunk_id,
                    "content": chunk_text,
                    "embedding": embedding,#.tolist(),
                    "metadata": chunk_metadata
                })
            
            # Save processed chunks to disk for potential reuse
            self._save_processed_chunks(doc_id, processed_chunks)
            
            return processed_chunks
        
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            raise
    
    def _create_medical_context_chunks(self, text: str) -> List[Tuple[str, str]]:
        """
        Create chunks that respect medical document structure.
        
        Args:
            text: Document text
            
        Returns:
            List of (chunk_text, section_name) tuples
        """
        chunks = []
        current_section = "unknown"
        
        # Find all section boundaries
        section_matches = list(self.section_pattern.finditer(text))
        
        if not section_matches:
            # If no sections found, fall back to standard chunking
            return self._fallback_chunking(text)
        
        # Process each section
        for i in range(len(section_matches)):
            start_pos = section_matches[i].start()
            section_name = text[section_matches[i].start():section_matches[i].end()].strip()
            
            # Determine section end
            if i < len(section_matches) - 1:
                end_pos = section_matches[i+1].start()
            else:
                end_pos = len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            # If section is too long, split into smaller chunks
            if len(section_text.split()) > self.chunk_size:
                section_chunks = self._chunk_text(section_text, section_name)
                chunks.extend(section_chunks)
            else:
                chunks.append((section_text, section_name))
        
        return chunks
    
    def _fallback_chunking(self, text: str) -> List[Tuple[str, str]]:
        """
        Fallback method for chunking when no sections are detected.
        
        Args:
            text: Document text
            
        Returns:
            List of (chunk_text, section_name) tuples
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append((chunk_text, "general"))
        
        return chunks
    
    def _chunk_text(self, text: str, section_name: str) -> List[Tuple[str, str]]:
        """
        Split text into chunks while trying to preserve sentence boundaries.
        
        Args:
            text: Text to chunk
            section_name: Name of the section
            
        Returns:
            List of (chunk_text, section_name) tuples
        """
        # Split by sentences (simplified - in production would use a better sentence tokenizer)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            # If adding this sentence exceeds chunk size and we already have content
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append((" ".join(current_chunk), section_name))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = len(current_chunk)
            
            # Add sentence to current chunk
            current_chunk.extend(sentence_words)
            current_length += sentence_length
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append((" ".join(current_chunk), section_name))
        
        return chunks
    
    def _extract_medical_entities(self, text: str) -> List[str]:
        """
        Extract medical entities from text using regex pattern.
        In production, this would be replaced with a medical NER model.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted medical entities
        """
        entities = set()
        for match in self.medical_entity_pattern.finditer(text.lower()):
            entities.add(match.group(0))
        
        return list(entities)
    
    def _save_processed_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]):
        """
        Save processed chunks to disk for potential reuse.
        
        Args:
            doc_id: Document identifier
            chunks: List of processed chunks
        """
        try:
            import json
            
            # Create filename
            filename = f"{doc_id}_processed.json"
            filepath = self.processed_docs_dir / filename
            
            # Save chunks without embeddings (to save space)
            chunks_without_embeddings = []
            for chunk in chunks:
                chunk_copy = chunk.copy()
                # Remove embedding as it's large and can be regenerated
                del chunk_copy["embedding"]
                chunks_without_embeddings.append(chunk_copy)
            
            with open(filepath, 'w') as f:
                json.dump(chunks_without_embeddings, f)
            
            self.logger.info(f"Saved processed chunks to {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save processed chunks: {e}")
    
    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of dictionaries with 'content' and 'metadata' keys
            
        Returns:
            List of processed document chunks with embeddings
        """
        all_processed_chunks = []
        
        for doc in documents:
            processed_chunks = self.process_document(doc["content"], doc["metadata"])
            all_processed_chunks.extend(processed_chunks)
        
        return all_processed_chunks
