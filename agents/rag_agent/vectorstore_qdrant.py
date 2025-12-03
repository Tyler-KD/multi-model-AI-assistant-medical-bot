import os
import re
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, OptimizersConfigDiff

class VectorStore:
    """
    Create vector store, ingest documents, retrieve relevant documents
    """
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.vector_search_type = config.rag.vector_search_type
        self.vectorstore_local_path = config.rag.vector_local_path
        self.docstore_local_path = config.rag.doc_local_path

        # Use the singleton client instead of creating a new one
        # self.client = QdrantClientManager.get_client(config)
        self.client = QdrantClient(path=self.vectorstore_local_path)

    def _does_collection_exist(self) -> bool:
        """Check if the collection already exists in Qdrant."""
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return self.collection_name in collection_names
        except Exception as e:
            self.logger.error(f"Error checking for collection existence: {e}")
            return False

    def _create_collection(self):
        """Create a new collection with dense and sparse vectors."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise e
            
    def load_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
        """
        Load existing vectorstore and docstore for retrieval operations without ingesting new documents.
        
        Returns:
            Tuple containing (vectorstore, docstore)
        """
        # Check if collection exists
        if not self._does_collection_exist():
            self.logger.error(f"Collection {self.collection_name} does not exist. Please ingest documents first.")
            raise ValueError(f"Collection {self.collection_name} does not exist")
            
        # Setup sparse embeddings
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Initialize vector store
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Document storage
        docstore = LocalFileStore(self.docstore_local_path)
        
        self.logger.info(f"Successfully loaded existing vectorstore and docstore")
        return qdrant_vectorstore, docstore

    def create_vectorstore(
            self,
            document_chunks: List[str],
            document_path: str,
        ) -> Tuple[QdrantVectorStore, LocalFileStore, List[str]]:
        """
        Create a vector store from document chunks or upsert documents to existing store.
        
        Args:
            document_chunks: List of document chunks
            document_path: Path to the original document
            
        Returns:
            Tuple containing (vectorstore, docstore, doc_ids)
        """
        
        # Generate unique IDs for each chunk
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create langchain documents
        langchain_documents = []
        for id_idx, chunk in enumerate(document_chunks):
            langchain_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(document_path),
                        "doc_id": doc_ids[id_idx],
                        # "source_path": Path(os.path.abspath(document_path)).as_uri()
                        "source_path": os.path.join("http://localhost:8000/", document_path)
                    }
                )
            )
        
        # Setup sparse embeddings
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Check if collection exists, create if it doesn't
        collection_exists = self._does_collection_exist()
        if not collection_exists:
            self._create_collection()
            self.logger.info(f"Created new collection: {self.collection_name}")
        else:
            self.logger.info(f"Collection {self.collection_name} already exists, will upsert documents")
        
        # Initialize vector store
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Document storage for parent documents
        docstore = LocalFileStore(self.docstore_local_path)
        
        # Ingest documents into vector and doc stores
        qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)
        
        # Encode string chunks to bytes before storing
        encoded_chunks = [chunk.encode('utf-8') for chunk in document_chunks]
        docstore.mset(list(zip(doc_ids, encoded_chunks)))

    def retrieve_relevant_chunks(
            self,
            query: str,
            vectorstore: QdrantVectorStore,
            docstore: LocalFileStore,
        ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve relevant chunks based on a query.
        
        Args:
            query: User query
            vectorstore: Vector store containing embeddings
            docstore: Document store containing actual content
            
        Returns:
            Tuple containing (retrieved_docs, picture_reference_paths)
            where retrieved_docs is a list of dictionaries with content and score
        """
        # Use similarity_search_with_score to get documents and scores
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k
        )
        
        retrieved_docs = []
        # picture_reference_paths = []
        
        for chunk, score in results:
            # Get full document from doc store as bytes and decode to string
            doc_content_bytes = docstore.mget([chunk.metadata['doc_id']])[0]
            doc_content = doc_content_bytes.decode('utf-8')
            
            # Add metadata to the document
            # formatted_doc = f"{doc_content}\nFollowing are the 'filename' and 'path as uri' of the source document for the current chunk: {chunk.metadata['source']}, {chunk.metadata['source_path']}"
            formatted_doc = doc_content
            
            # Create document dict in the format expected by reranker
            doc_dict = {
                "id": chunk.metadata['doc_id'],
                "content": formatted_doc,
                "score": score,  # Use the actual similarity score
                "source": chunk.metadata['source'],
                "source_path": chunk.metadata['source_path'],
            }
            retrieved_docs.append(doc_dict)
            
            # # Extract picture references
            # matches = re.finditer(r"picture_counter_(\d+)", doc_content)
            # for match in matches:
            #     counter_value = int(match.group(1))
            #     # Create picture path based on document source and counter
            #     doc_basename = os.path.splitext(chunk.metadata['source'])[0]  # Remove file extension
            #     picture_path = Path(os.path.abspath(parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")).as_uri()
            #     picture_reference_paths.append(picture_path)
        
        # return retrieved_docs, picture_reference_paths
        return retrieved_docs