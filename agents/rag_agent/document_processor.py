import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import os
from pathlib import Path
import hashlib
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import numpy as np
import json

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class MedicalDocumentProcessor:
    """
    Advanced processor for various medical documents with multiple chunking strategies.
    """
    def __init__(self, config, embedding_model):
        """
        Initialize the document processor with configurable chunking strategies.
        
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
        
        # Chunking strategy selection
        self.chunking_strategy = getattr(config.rag, "chunking_strategy", "hybrid")
        self.logger.info(f"Using chunking strategy: {self.chunking_strategy}")
        
        # Document type detection patterns - Fixed by moving the flags to the beginning
        self.document_type_patterns = {
            "research_paper": re.compile(r"(?i)(abstract|introduction|methods|results|discussion|conclusion|references)"),
            "clinical_note": re.compile(r"(?i)(chief complaint|history of present illness|past medical history|medications|assessment|plan)"),
            "patient_record": re.compile(r"(?i)(patient information|vital signs|allergies|family history|social history)"),
            "medical_guideline": re.compile(r"(?i)(recommendations|guidelines|protocols|indications|contraindications)"),
            "drug_information": re.compile(r"(?i)(mechanism of action|pharmacokinetics|dosage|side effects|interactions)")
        }
        
        # Medical section headers - Fixed by using separate re.IGNORECASE flag instead of inline (?i)
        self.section_headers = [
            # Research papers
            r"^(abstract|introduction|background|methods?|results?|discussion|conclusion|references)",
            
            # Clinical notes
            r"^(chief complaint|history of present illness|hpi|past medical history|pmh|"
            r"medications|assessment|plan|review of systems|ros|physical examination|"
            r"lab results|imaging|impression|followup)",
            
            # Patient records
            r"^(patient information|demographics|vital signs|allergies|immunizations|"
            r"family history|social history|surgical history|problem list)",
            
            # Medical conditions
            r"^(clinical presentation|diagnosis|treatment|prognosis|etiology|"
            r"epidemiology|pathophysiology|signs and symptoms|complications|prevention|"
            r"patient education|differential diagnosis)",
            
            # Guidelines and protocols
            r"^(recommendations|guidelines|protocols|indications|contraindications|"
            r"dosage|administration|monitoring|special populations)",
            
            # Drug information
            r"^(mechanism of action|pharmacokinetics|pharmacodynamics|dosing|"
            r"adverse effects|warnings|interactions|storage|pregnancy considerations)"
        ]
        
        filtered_headers = [header for header in self.section_headers if header.strip()]
        self.section_pattern = re.compile(f"({'|'.join(filtered_headers)})", re.IGNORECASE)
        
        # Enhanced medical entities detection - Fixed by using separate re.IGNORECASE flag
        # This would ideally be replaced with a proper medical NER model in production
        self.medical_entity_categories = {
            "diseases": r"(diabetes|hypertension|cancer|asthma|covid-19|stroke|"
                      r"alzheimer's|parkinson's|arthritis|obesity|heart disease|hepatitis|"
                      r"influenza|pneumonia|tuberculosis|hiv/aids|malaria|cholera|"
                      r"diabetes mellitus|chronic kidney disease|copd)",
            
            "medications": r"(aspirin|ibuprofen|acetaminophen|lisinopril|metformin|"
                         r"atorvastatin|omeprazole|amoxicillin|prednisone|insulin|"
                         r"albuterol|levothyroxine|warfarin|clopidogrel|metoprolol)",
            
            "procedures": r"(surgery|biopsy|endoscopy|colonoscopy|mri|ct scan|x-ray|"
                        r"ultrasound|echocardiogram|ekg|ecg|angiography|mammography|"
                        r"vaccination|immunization|blood test|urinalysis)",
            
            "anatomy": r"(heart|lung|liver|kidney|brain|stomach|intestine|colon|"
                     r"pancreas|spleen|thyroid|adrenal|pituitary|bone|muscle|nerve|"
                     r"artery|vein|capillary|joint|skin)"
        }
        
        # Combine all entity patterns
        all_patterns = []
        for category, pattern in self.medical_entity_categories.items():
            all_patterns.append(f"(?P<{category}>{pattern})")
        
        self.medical_entity_pattern = re.compile("|".join(all_patterns), re.IGNORECASE)
        
    def process_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a document using the selected chunking strategy.
        
        Args:
            content: Document content string
            metadata: Document metadata including source, specialty, etc.
            
        Returns:
            List of processed document chunks with embeddings
        """
        try:
            # Detect document type
            doc_type = self._detect_document_type(content)
            
            # Create document ID based on content hash
            doc_id_base = hashlib.md5(content.encode()).hexdigest()
            doc_id = str(uuid.UUID(doc_id_base[:32]))
            
            # Extract medical entities
            medical_entities = self._extract_medical_entities(content)
            
            # Add entities and document type to metadata
            enhanced_metadata = metadata.copy()
            enhanced_metadata['medical_entities'] = medical_entities
            enhanced_metadata['document_type'] = doc_type
            enhanced_metadata['processing_timestamp'] = datetime.now().isoformat()
            
            # Create chunks based on the selected strategy
            if self.chunking_strategy == "semantic":
                chunks = self._create_semantic_chunks(content, doc_type)
            elif self.chunking_strategy == "sliding_window":
                chunks = self._create_sliding_window_chunks(content)
            elif self.chunking_strategy == "recursive":
                chunks = self._create_recursive_chunks(content)
            elif self.chunking_strategy == "hybrid":
                chunks = self._create_hybrid_chunks(content, doc_type)
            else:
                # Default to hybrid method
                chunks = self._create_hybrid_chunks(content, doc_type)
            
            # Process each chunk
            processed_chunks = []
            for i, chunk_info in enumerate(chunks):
                if isinstance(chunk_info, tuple):
                    chunk_text, section, level = chunk_info[0], chunk_info[1], chunk_info[2] if len(chunk_info) > 2 else "standard"
                else:
                    chunk_text, section, level = chunk_info, "general", "standard"
                
                # Generate chunk ID as a UUID with a suffix
                chunk_id = str(uuid.UUID(doc_id_base[:24] + f"{i:08}"))
                
                # Calculate chunk importance score based on entity density and position
                importance_score = self._calculate_chunk_importance(chunk_text, i, len(chunks))
                
                # Generate embedding
                embedding = self.embedding_model.embed_documents([chunk_text])[0]
                
                # Create chunk metadata
                chunk_metadata = enhanced_metadata.copy()
                chunk_metadata["chunk_number"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["section"] = section
                chunk_metadata["hierarchy_level"] = level
                chunk_metadata["importance_score"] = importance_score
                chunk_metadata["word_count"] = len(chunk_text.split())
                chunk_metadata["chunking_strategy"] = self.chunking_strategy
                
                # Add related chunks for context linkage
                if i > 0:
                    chunk_metadata["previous_chunk_id"] = str(uuid.UUID(doc_id_base[:24] + f"{i-1:08}"))
                if i < len(chunks) - 1:
                    chunk_metadata["next_chunk_id"] = str(uuid.UUID(doc_id_base[:24] + f"{i+1:08}"))
                
                # Create processed chunk
                processed_chunks.append({
                    "id": chunk_id,
                    "content": chunk_text,
                    "embedding": embedding,
                    "metadata": self.make_serializable(chunk_metadata)
                })
            
            # Save processed chunks to disk for potential reuse
            self._save_processed_chunks(doc_id, processed_chunks)
            
            return processed_chunks
        
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            raise
    
    def _detect_document_type(self, text: str) -> str:
        """
        Detect the type of medical document based on content patterns.
        
        Args:
            text: Document text
            
        Returns:
            Document type string
        """
        type_scores = {}
        
        # Check each document type pattern
        for doc_type, pattern in self.document_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[doc_type] = len(matches)
        
        # Find the document type with the highest number of matches
        if max(type_scores.values(), default=0) > 0:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        # Default to general if no clear type
        return "general_medical"
    
    def _create_semantic_chunks(self, text: str, doc_type: str) -> List[Tuple[str, str]]:
        """
        Create chunks that respect semantic boundaries in the document.
        
        Args:
            text: Document text
            doc_type: Document type
            
        Returns:
            List of (chunk_text, section_name) tuples
        """
        # Find all section boundaries
        section_matches = list(self.section_pattern.finditer(text))
        chunks = []
        
        if not section_matches:
            # If no sections found, fall back to paragraph-based chunking
            paragraphs = re.split(r'\n\s*\n', text)
            for i, para in enumerate(paragraphs):
                if para.strip():
                    chunks.append((para.strip(), "paragraph", "standard"))
            return chunks
        
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
            
            # Split section into paragraphs if it's too large
            if len(section_text.split()) > self.chunk_size:
                section_chunks = self._split_into_paragraphs(section_text, section_name)
                chunks.extend(section_chunks)
            else:
                chunks.append((section_text, section_name, "section"))
        
        return chunks
    
    def _split_into_paragraphs(self, text: str, section_name: str) -> List[Tuple[str, str, str]]:
        """
        Split text into paragraph-level chunks.
        
        Args:
            text: Text to split
            section_name: Name of the section
            
        Returns:
            List of (chunk_text, section_name, level) tuples
        """
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
                
            # Check if paragraph is too large
            if len(para.split()) > self.chunk_size:
                # Further split into sentences
                sentences = sent_tokenize(para)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence.split())
                    
                    if current_length + sentence_length > self.chunk_size and current_chunk:
                        # Add current chunk
                        chunk_text = " ".join(current_chunk)
                        chunks.append((chunk_text, section_name, "paragraph"))
                        current_chunk = []
                        current_length = 0
                    
                    current_chunk.append(sentence)
                    current_length += sentence_length
                
                # Add final chunk if not empty
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, section_name, "paragraph"))
            else:
                chunks.append((para.strip(), section_name, "paragraph"))
        
        return chunks
    
    def _create_sliding_window_chunks(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Create overlapping chunks using a sliding window approach.
        
        Args:
            text: Document text
            
        Returns:
            List of (chunk_text, section_name, level) tuples
        """
        sentences = sent_tokenize(text)
        chunks = []
        
        # If very few sentences, return as one chunk
        if len(sentences) <= 3:
            return [(text, "full_document", "document")]
        
        # Calculate stride (number of sentences to slide window)
        stride = max(1, (self.chunk_size - self.chunk_overlap) // 20)  # Approximate words per sentence
        
        # Create chunks with sliding window
        for i in range(0, len(sentences), stride):
            # Determine end index for current window
            window_size = min(i + max(3, self.chunk_size // 20), len(sentences))
            
            # Get text for current window
            window_text = " ".join(sentences[i:window_size])
            
            # Detect current section if possible
            section_match = self.section_pattern.search(window_text)
            section_name = section_match.group(0) if section_match else "sliding_window"
            
            chunks.append((window_text, section_name, "sliding"))
        
        return chunks
    
    def _create_recursive_chunks(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Create hierarchical chunks at different levels of granularity.
        
        Args:
            text: Document text
            
        Returns:
            List of (chunk_text, section_name, level) tuples
        """
        chunks = []
        
        # Level 1: Document-level chunk (if not too large)
        if len(text.split()) <= self.chunk_size * 2:
            chunks.append((text, "full_document", "document"))
        
        # Level 2: Section-level chunks
        section_matches = list(self.section_pattern.finditer(text))
        
        if section_matches:
            for i in range(len(section_matches)):
                start_pos = section_matches[i].start()
                section_name = text[section_matches[i].start():section_matches[i].end()].strip()
                
                # Determine section end
                if i < len(section_matches) - 1:
                    end_pos = section_matches[i+1].start()
                else:
                    end_pos = len(text)
                
                section_text = text[start_pos:end_pos].strip()
                
                # Add section as a chunk
                if section_text and len(section_text.split()) <= self.chunk_size:
                    chunks.append((section_text, section_name, "section"))
                
                # Level 3: Paragraph-level chunks
                paragraphs = re.split(r'\n\s*\n', section_text)
                
                for j, para in enumerate(paragraphs):
                    if para.strip() and len(para.split()) <= self.chunk_size:
                        chunks.append((para.strip(), section_name, "paragraph"))
                    
                    # Level 4: Sentence-level chunks for important sentences
                    if self._contains_important_entities(para):
                        sentences = sent_tokenize(para)
                        for sentence in sentences:
                            if self._contains_important_entities(sentence):
                                chunks.append((sentence.strip(), section_name, "sentence"))
        else:
            # No clear sections, fall back to paragraphs and sentences
            paragraphs = re.split(r'\n\s*\n', text)
            
            for para in paragraphs:
                if para.strip() and len(para.split()) <= self.chunk_size:
                    chunks.append((para.strip(), "paragraph", "paragraph"))
        
        return chunks
    
    def _create_hybrid_chunks(self, text: str, doc_type: str) -> List[Tuple[str, str, str]]:
        """
        Create chunks using a hybrid approach that adapts to document type.
        
        Args:
            text: Document text
            doc_type: Detected document type
            
        Returns:
            List of (chunk_text, section_name, level) tuples
        """
        chunks = []
        
        # First identify sections
        section_matches = list(self.section_pattern.finditer(text))
        
        # If the document has clear sections
        if section_matches:
            sections = []
            
            # Extract all sections
            for i in range(len(section_matches)):
                start_pos = section_matches[i].start()
                section_name = text[section_matches[i].start():section_matches[i].end()].strip()
                
                # Determine section end
                if i < len(section_matches) - 1:
                    end_pos = section_matches[i+1].start()
                else:
                    end_pos = len(text)
                
                section_text = text[start_pos:end_pos].strip()
                sections.append((section_text, section_name))
            
            # Process each section based on its content characteristics
            for section_text, section_name in sections:
                # Determine content complexity based on medical entity density
                entity_density = len(self._extract_medical_entities(section_text)) / max(1, len(section_text.split()) / 100)
                
                # Adapt chunk size based on content complexity
                adaptive_chunk_size = self.chunk_size
                if entity_density > 0.5:  # High entity density
                    adaptive_chunk_size = int(self.chunk_size * 0.7)  # Smaller chunks for dense content
                
                # If the section is a summary section (abstract, conclusion), keep it whole if possible
                if re.search(r"(?i)(abstract|summary|conclusion)", section_name) and len(section_text.split()) <= adaptive_chunk_size:
                    chunks.append((section_text, section_name, "key_section"))
                    continue
                
                # For other sections, split based on their size
                if len(section_text.split()) <= adaptive_chunk_size:
                    chunks.append((section_text, section_name, "section"))
                else:
                    # Split into paragraphs or sentences as needed
                    paragraphs = re.split(r'\n\s*\n', section_text)
                    
                    if len(paragraphs) <= 1 or doc_type == "clinical_note":
                        # For clinical notes or single-paragraph sections, use sentence-based chunking
                        chunks.extend(self._chunk_by_sentences(section_text, section_name, adaptive_chunk_size))
                    else:
                        # For multi-paragraph sections, process each paragraph
                        for para in paragraphs:
                            if not para.strip():
                                continue
                                
                            if len(para.split()) <= adaptive_chunk_size:
                                chunks.append((para.strip(), section_name, "paragraph"))
                            else:
                                # For long paragraphs, split by sentences
                                chunks.extend(self._chunk_by_sentences(para, section_name, adaptive_chunk_size))
        else:
            # For documents without clear sections, use a mix of paragraph and sliding window
            if doc_type in ["clinical_note", "patient_record"]:
                # Clinical notes often have implicit structure without formal headers
                chunks.extend(self._create_sliding_window_chunks(text))
            else:
                # For other types, try paragraph-based chunking
                paragraphs = re.split(r'\n\s*\n', text)
                
                if len(paragraphs) <= 1:
                    # If it's essentially one big paragraph, use sliding window
                    chunks.extend(self._create_sliding_window_chunks(text))
                else:
                    # Process each paragraph
                    for para in paragraphs:
                        if not para.strip():
                            continue
                            
                        if len(para.split()) <= self.chunk_size:
                            chunks.append((para.strip(), "paragraph", "paragraph"))
                        else:
                            # For long paragraphs, use sentence chunking
                            chunks.extend(self._chunk_by_sentences(para, "paragraph", self.chunk_size))
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, section_name: str, chunk_size: int) -> List[Tuple[str, str, str]]:
        """
        Create chunks by grouping sentences while respecting chunk size.
        
        Args:
            text: Text to chunk
            section_name: Name of the section
            chunk_size: Maximum chunk size in words
            
        Returns:
            List of (chunk_text, section_name, level) tuples
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            # If adding this sentence exceeds chunk size and we already have content
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, section_name, "sentences"))
                
                # Start new chunk with overlap
                # Find a good overlap point that doesn't split mid-thought
                overlap_sentences = min(2, len(current_chunk))
                current_chunk = current_chunk[-overlap_sentences:]
                current_length = len(" ".join(current_chunk).split())
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk if not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, section_name, "sentences"))
        
        return chunks
    
    def _contains_important_entities(self, text: str) -> bool:
        """
        Check if text contains important medical entities.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating presence of important entities
        """
        entities = self._extract_medical_entities(text)
        return len(entities) > 0
    
    def _calculate_chunk_importance(self, text: str, position: int, total_chunks: int) -> float:
        """
        Calculate importance score for a chunk based on various factors.
        
        Args:
            text: Chunk text
            position: Position in document
            total_chunks: Total number of chunks
            
        Returns:
            Importance score between 0 and 1
        """
        # Extract entities and count them
        entities = self._extract_medical_entities(text)
        entity_count = len(entities)
        
        # Calculate entity density
        word_count = len(text.split())
        entity_density = entity_count / max(1, word_count / 100)
        
        # Position importance - first and last chunks often contain key information
        position_score = 0.0
        if position == 0 or position == total_chunks - 1:
            position_score = 0.2
        elif position < total_chunks * 0.2 or position > total_chunks * 0.8:
            position_score = 0.1
        
        # Check for important keywords
        keyword_score = 0.0
        important_keywords = ["significant", "important", "critical", "essential", "key", 
                             "finding", "diagnosis", "recommend", "conclude", "summary"]
        for keyword in important_keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE):
                keyword_score += 0.05
        keyword_score = min(0.2, keyword_score)
        
        # Combine scores
        importance_score = min(1.0, 0.3 * entity_density + position_score + keyword_score)
        
        return importance_score
    
    def _extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text by category.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of categorized medical entities
        """
        categorized_entities = {}
        
        for category, pattern in self.medical_entity_categories.items():
            category_pattern = re.compile(pattern)
            matches = set(m.group(0).lower() for m in category_pattern.finditer(text))
            if matches:
                categorized_entities[category] = list(matches)
        
        return categorized_entities
    
    def _save_processed_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]):
        """
        Save processed chunks to disk for potential reuse.
        
        Args:
            doc_id: Document identifier
            chunks: List of processed chunks
        """
        try:
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
            try:
                processed_chunks = self.process_document(doc["content"], doc["metadata"])
                all_processed_chunks.extend(processed_chunks)
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                # Continue with the next document
                continue
        
        return all_processed_chunks
    
    def make_serializable(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert non-serializable objects in metadata to a serializable format.
        
        Args:
            metadata: Metadata dictionary.
        
        Returns:
            A JSON-serializable metadata dictionary.
        """
        serializable_metadata = {}
        for key, value in metadata.items():
            try:
                json.dumps(value)  # Test if value is serializable
                serializable_metadata[key] = value
            except TypeError:
                # Handle non-serializable objects
                if isinstance(value, Header):
                    serializable_metadata[key] = str(value)  # Convert Header to string
                else:
                    serializable_metadata[key] = str(value)  # Convert other non-serializable objects to string
        return serializable_metadata

class Header:
    def __init__(self, text):
        self.text = text
    
    def __str__(self):
        return self.text