import os
import time
import logging
from typing import List, Optional, Dict, Any

from .doc_parser import MedicalDocParser
from .content_processor import ContentProcessor
from .vectorstore_qdrant import VectorStore
from .reranker import Reranker
from .query_expander import QueryExpander
from .response_generator import ResponseGenerator

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
        # Set up logging
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing Medical RAG system")
        self.config = config
        self.doc_parser = MedicalDocParser()
        self.content_processor = ContentProcessor(config)
        self.vector_store = VectorStore(config)
        self.reranker = Reranker(config)
        self.query_expander = QueryExpander(config)
        self.response_generator = ResponseGenerator(config)
        self.parsed_content_dir = self.config.rag.parsed_content_dir
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all files in a directory into the RAG system.
        
        Args:
            directory_path: Path to the directory containing files to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting files from directory: {directory_path}")
        
        try:
            # Check if directory exists
            if not os.path.isdir(directory_path):
                raise ValueError(f"Directory not found: {directory_path}")
            
            # Get all files in the directory
            files = [os.path.join(directory_path + '/', f) for f in os.listdir(directory_path) 
                     if os.path.isfile(os.path.join(directory_path, f))]
            
            if not files:
                self.logger.warning(f"No files found in directory: {directory_path}")
                return {
                    "success": True,
                    "documents_ingested": 0,
                    "chunks_processed": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Track statistics
            total_chunks_processed = 0
            successful_ingestions = 0
            failed_ingestions = 0
            failed_files = []
            
            # Process each file
            for file_path in files:
                self.logger.info(f"Processing file {successful_ingestions + failed_ingestions + 1}/{len(files)}: {file_path}")
                
                try:
                    result = self.ingest_file(file_path)
                    if result["success"]:
                        successful_ingestions += 1
                        total_chunks_processed += result.get("chunks_processed", 0)
                    else:
                        failed_ingestions += 1
                        failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    failed_ingestions += 1
                    failed_files.append({"file": file_path, "error": str(e)})
            
            return {
                "success": True,
                "documents_ingested": successful_ingestions,
                "failed_documents": failed_ingestions,
                "failed_files": failed_files,
                "chunks_processed": total_chunks_processed,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def ingest_file(self, document_path: str) -> Dict[str, Any]:
        """
        Ingest a single file into the RAG system.
        
        Args:
            document_path: Path to the file to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting file: {document_path}")

        try:
            # Step 1: Parse document
            self.logger.info("1. Parsing document and extracting images...")
            parsed_document, images = self.doc_parser.parse_document(document_path, self.parsed_content_dir)
            self.logger.info(f"   Parsed document and extracted {len(images)} images")

            # Step 2: Summarize images
            self.logger.info("2. Summarizing images...")
            image_summaries = self.content_processor.summarize_images(images)
            self.logger.info(f"   Generated {len(image_summaries)} image summaries")

            # Step 3: Format document with image summaries
            self.logger.info("3. Formatting document with image summaries...")
            formatted_document = self.content_processor.format_document_with_images(parsed_document, image_summaries)

            # Step 4: Chunk document into semantic sections
            self.logger.info("4. Chunking document into semantic sections...")
            document_chunks = self.content_processor.chunk_document(formatted_document)
            self.logger.info(f"   Document split into {len(document_chunks)} chunks")

            # Step 5: Create vector store and document store
            self.logger.info("5. Creating vector store knowledge base...")
            self.vector_store.create_vectorstore(
                document_chunks=document_chunks, 
                document_path=document_path
                )
            
            return {
                "success": True,
                "documents_ingested": 1,
                "chunks_processed": len(document_chunks),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            self.logger.error(f"Error ingesting file: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a query with the RAG system.
        
        Args:
            query: The query string
            chat_history: Optional chat history for context
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        self.logger.info(f"RAG Agent processing query: {query}")
        
        # Process query and return result, passing chat_history
        try:
            # Step 1: Expand query
            self.logger.info(f"1. Expanding query: '{query}'")
            expansion_result = self.query_expander.expand_query(query)
            expanded_query = expansion_result["expanded_query"]
            self.logger.info(f"   Original: '{query}'")
            self.logger.info(f"   Expanded: '{expanded_query}'")
            query = expanded_query

            # Step 2: Retrieval
            self.logger.info(f"2. Retrieving relevant documents for the query: '{query}'")
            vectorstore, docstore = self.vector_store.load_vectorstore()
            retrieved_documents = self.vector_store.retrieve_relevant_chunks(
                query=query,
                vectorstore=vectorstore,
                docstore=docstore,
                )

            self.logger.info(f"   Retrieved {len(retrieved_documents)} relevant document chunks")

            # Step 3: Rerank the retrieved documents if we have a reranker and enough documents
            self.logger.info(f"3. Reranking the retrieved documents")
            if self.reranker and len(retrieved_documents) > 1:
                reranked_documents, reranked_top_k_picture_paths = self.reranker.rerank(query, retrieved_documents, self.parsed_content_dir)
                self.logger.info(f"   Reranked retrieved documents and chose top {len(reranked_documents)}")
                self.logger.info(f"   Found {len(reranked_top_k_picture_paths)} referenced images")
            else:
                self.logger.info(f"   Could not rerank the retrieved documents, falling back to original scores")
                reranked_documents = retrieved_documents
                reranked_top_k_picture_paths = []

            # Step 4: Generate response
            self.logger.info("4. Generating response...")
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=reranked_documents,
                picture_paths=reranked_top_k_picture_paths,
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