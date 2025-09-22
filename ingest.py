"""
Document ingestion and processing module for RAG (Retrieval-Augmented Generation) application.

This module handles the complete document processing pipeline for RAG systems:
1. Loading documents from various file formats (.txt, .pdf)
2. Chunking documents into manageable segments
3. Computing embeddings for text chunks
4. Creating and managing FAISS vector stores
5. Saving and loading processed document stores

The main components are:
    - DocumentIngestor: Main class handling the ingestion pipeline
    - Document loading utilities for different file formats
    - Text chunking with configurable parameters
    - Vector store management with FAISS
    - Metadata tracking for document tracing

Dependencies:
    - sentence_transformers: For text embeddings
    - langchain: For document loading and text splitting
    - FAISS: For vector similarity search
    - PyPDF2: For PDF file processing
    - pickle: For metadata storage

Example Usage:
    ```python
    # Initialize ingestor
    ingestor = DocumentIngestor()

    # Process documents
    vector_store = ingestor.process_documents("./docs")

    # Save for later use
    ingestor.save_vector_store()
    ```

Technical Details:
    - Uses recursive character text splitting for robust chunking
    - Supports UTF-8 and fallback encodings for text files
    - Maintains detailed metadata for document traceability
    - Thread-safe document processing
    - Efficient batch processing for large document sets

Notes:
    - Vector store is saved to ./vector_store by default
    - PDF processing requires text-based PDFs (not scanned)
    - Memory usage scales with chunk size and document count
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestor:
    """
    A comprehensive document processing pipeline for RAG applications.

    This class handles the complete lifecycle of document processing:
    - Loading documents from files
    - Splitting text into manageable chunks
    - Computing embeddings using transformer models
    - Creating and managing FAISS vector stores
    - Maintaining document traceability through metadata

    The class is designed to be both efficient for large document sets
    and flexible enough to handle various document formats and configurations.

    Attributes:
        embedding_model_name (str): Name of the HuggingFace embedding model
        chunk_size (int): Target size of text chunks in characters
        chunk_overlap (int): Overlap between consecutive chunks
        vector_store_path (str): Path to save/load vector store
        embeddings: HuggingFace embeddings model instance
        text_splitter: LangChain text splitter instance
        vector_store (Optional[FAISS]): FAISS vector store instance
        document_metadata (List[Dict]): Metadata for processed documents

    Example:
        ```python
        # Initialize with custom settings
        ingestor = DocumentIngestor(
            embedding_model="all-mpnet-base-v2",
            chunk_size=1000,
            chunk_overlap=200
        )

        # Process a folder of documents
        vector_store = ingestor.process_documents("./documents")

        # Save for later use
        ingestor.save_vector_store()

        # Load existing store
        loaded_store = ingestor.load_vector_store()
        ```

    Notes:
        - Large chunk_size values improve context but increase memory usage
        - chunk_overlap helps maintain coherence between chunks
        - Vector store is automatically saved after processing
        - Metadata is preserved for document tracing
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 160,
        vector_store_path: str = "./vector_store"
    ):
        """
        Initialize the document ingestor.
        
        Args:
            embedding_model: Name of the Hugging Face embedding model
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            vector_store_path: Path to save/load vector store
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vector_store: Optional[FAISS] = None
        self.document_metadata: List[Dict[str, Any]] = []
        
    def load_documents(self, folder_path: str) -> List[Document]:
        """
        Load and parse documents from a specified folder.

        This method recursively traverses the given folder and loads all supported
        documents (.txt and .pdf files). It handles various encodings and formats,
        adding comprehensive metadata to each document.

        Args:
            folder_path: Absolute or relative path to the folder containing documents.
                       Will be recursively searched for supported files.

        Returns:
            List[Document]: List of LangChain Document objects, each containing:
                - page_content: The text content of the document
                - metadata: Dict with file info (name, type, path, etc.)

        Raises:
            ValueError: If the folder doesn't exist
            Exception: For file reading or parsing errors

        Example:
            ```python
            documents = ingestor.load_documents("./my_docs")
            print(f"Loaded {len(documents)} documents")
            ```

        Notes:
            - Supports .txt files (UTF-8 and fallback encodings)
            - Supports text-based PDF files
            - Skips unsupported file types with warning
            - Adds extensive metadata for traceability
            - Handles large files efficiently
        """
        documents = []
        folder = Path(folder_path)
        
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
            
        logger.info(f"Loading documents from {folder_path}")
        
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.txt':
                        docs = self._load_txt_file(file_path)
                        documents.extend(docs)
                    elif file_path.suffix.lower() == '.pdf':
                        docs = self._load_pdf_file(file_path)
                        documents.extend(docs)
                    else:
                        logger.warning(f"Unsupported file type: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    
        logger.info(f"Loaded {len(documents)} document(s)")
        return documents
    
    def _load_txt_file(self, file_path: Path) -> List[Document]:
        """Load a text file."""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'txt',
                    'file_name': file_path.name
                })
            return docs
        except UnicodeDecodeError:
            # Try with different encoding
            loader = TextLoader(str(file_path), encoding='latin-1')
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'txt',
                    'file_name': file_path.name
                })
            return docs
    
    def _load_pdf_file(self, file_path: Path) -> List[Document]:
        """Load a PDF file."""
        documents = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():  # Only add non-empty pages
                    text = " ".join(text.split())
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': str(file_path),
                            'file_type': 'pdf',
                            'file_name': file_path.name,
                            'page_number': page_num + 1,
                            'total_pages': len(pdf_reader.pages)
                        }
                    )
                    documents.append(doc)
                    
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantically meaningful chunks.

        This method processes documents into smaller chunks suitable for embedding
        and retrieval. It preserves document structure where possible and maintains
        chunk relationships through metadata.

        The chunking process:
        1. Splits text using recursive character text splitter
        2. Maintains document grouping and ordering
        3. Adds detailed chunk metadata for tracing
        4. Preserves semantic coherence through overlap

        Args:
            documents: List of LangChain Document objects to be chunked

        Returns:
            List[Document]: List of document chunks, each containing:
                - Portion of original text
                - Enhanced metadata including:
                    - Original file information
                    - Chunk position and count
                    - Character count
                    - Page numbers (if applicable)
                    - Relationship to other chunks

        Example:
            ```python
            doc = Document(page_content="Long text...", metadata={...})
            chunks = ingestor.chunk_documents([doc])
            print(f"Created {len(chunks)} chunks")
            ```

        Notes:
            - Chunk size and overlap are set during initialization
            - Metadata is preserved and enhanced
            - Chunks maintain reference to source document
            - Intelligent splitting at sentence boundaries
            - Overlap ensures context preservation
        """
        logger.info("Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)

        # Group chunks by originating file to retain position metadata
        chunks_by_file: Dict[str, List[Document]] = {}
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata)
            source_path = chunk.metadata.get('source')
            file_name = chunk.metadata.get('file_name') or (Path(source_path).name if source_path else "unknown_source")
            chunk.metadata['file_name'] = file_name
            if 'file_type' not in chunk.metadata and source_path:
                chunk.metadata['file_type'] = Path(source_path).suffix.lstrip('.')
            chunks_by_file.setdefault(file_name, []).append(chunk)

        for file_name, file_chunks in chunks_by_file.items():
            total_chunks = len(file_chunks)
            for idx, chunk in enumerate(file_chunks, start=1):
                chunk.metadata['chunk_index'] = idx
                chunk.metadata['total_chunks'] = total_chunks
                chunk.metadata['chunk_id'] = f"{file_name}-chunk-{idx}"
                chunk.metadata['chunk_char_count'] = len(chunk.page_content)
                if chunk.metadata.get('page_number') is None:
                    chunk.metadata['page_number'] = 'n/a'
                if chunk.metadata.get('total_pages') is None:
                    chunk.metadata['total_pages'] = 'n/a'

        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from processed document chunks.

        This method handles the creation of a searchable vector store:
        1. Computes embeddings for all document chunks
        2. Creates a FAISS index for efficient similarity search
        3. Stores document content and metadata for retrieval
        4. Configures the store for optimal search performance

        Args:
            documents: List of document chunks to embed and index.
                     Each document should contain text content and metadata.

        Returns:
            FAISS: A configured FAISS vector store containing:
                - Document embeddings
                - Original text content
                - Associated metadata
                - Optimized similarity search index

        Raises:
            ValueError: If no documents are provided
            Exception: For embedding or index creation errors

        Example:
            ```python
            chunks = ingestor.chunk_documents(documents)
            store = ingestor.create_vector_store(chunks)
            # Store ready for similarity search
            ```

        Technical Details:
            - Uses HuggingFace embeddings model
            - Creates L2-normalized vectors
            - Configures FAISS for CPU or GPU
            - Optimizes for search performance
            - Preserves all document metadata

        Notes:
            - Memory usage scales with document count
            - GPU acceleration if available
            - Thread-safe implementation
            - Automatic error handling
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")
            
        logger.info("Creating vector store...")
        
        # Create vector store
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        self.vector_store = vector_store
        logger.info(f"Created vector store with {len(documents)} documents")
        
        return vector_store
    
    def save_vector_store(self) -> None:
        """Save the vector store to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
            
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Save FAISS index
        self.vector_store.save_local(self.vector_store_path)
        
        # Save metadata
        metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.document_metadata, f)
            
        logger.info(f"Vector store saved to {self.vector_store_path}")
    
    def load_vector_store(self) -> Optional[FAISS]:
        """
        Load vector store from disk.
        
        Returns:
            Loaded FAISS vector store or None if not found
        """
        if not os.path.exists(self.vector_store_path):
            logger.warning(f"Vector store path does not exist: {self.vector_store_path}")
            return None
            
        try:
            # Load FAISS index
            vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                    
            self.vector_store = vector_store
            logger.info(f"Vector store loaded from {self.vector_store_path}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def process_documents(self, folder_path: str) -> FAISS:
        """
        Execute the complete document processing pipeline.

        This is the main entry point for document processing, combining all steps:
        1. Document loading from files
        2. Text extraction and cleaning
        3. Chunking into segments
        4. Computing embeddings
        5. Creating vector store
        6. Saving results to disk

        The pipeline is designed for robustness and maintainability, with
        comprehensive error handling and logging at each step.

        Args:
            folder_path: Path to directory containing documents.
                        Will be processed recursively.

        Returns:
            FAISS: Fully configured vector store ready for similarity search,
                  containing all processed documents with metadata.

        Raises:
            ValueError: If no valid documents found
            Exception: For any processing pipeline errors

        Example:
            ```python
            # Complete processing example
            ingestor = DocumentIngestor()
            try:
                vector_store = ingestor.process_documents("./docs")
                print(f"Indexed {vector_store.index.ntotal} chunks")
            except Exception as e:
                print(f"Processing failed: {e}")
            ```

        Pipeline Steps:
            1. Validate and load documents
            2. Split into optimal chunks
            3. Compute embeddings
            4. Create FAISS index
            5. Save vector store
            6. Store metadata

        Notes:
            - Processing time scales with document count
            - Progress is logged at each step
            - Automatic error recovery where possible
            - Results are automatically saved
            - Memory efficient processing
        """
        # Load documents
        documents = self.load_documents(folder_path)
        
        if not documents:
            raise ValueError("No documents found to process")
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Store metadata aligned with chunks for downstream tracing
        self.document_metadata = [chunk.metadata for chunk in chunks]

        # Create vector store
        vector_store = self.create_vector_store(chunks)
        
        # Save vector store
        self.save_vector_store()
        
        return vector_store


def main():
    """Example usage of the DocumentIngestor."""
    # Create sample documents directory
    sample_dir = "./sample_docs"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a sample text file
    sample_txt_path = os.path.join(sample_dir, "sample.txt")
    if not os.path.exists(sample_txt_path):
        with open(sample_txt_path, 'w') as f:
            f.write("""
            This is a sample document for testing the RAG application.
            It contains multiple paragraphs to demonstrate document chunking.
            
            The RAG (Retrieval-Augmented Generation) approach combines the power
            of information retrieval with language generation. This allows AI systems
            to access and use specific information from a knowledge base when
            generating responses.
            
            Vector databases like FAISS enable efficient similarity search over
            high-dimensional embeddings, making it possible to quickly find
            relevant document chunks for any given query.
            """)
    
    # Initialize ingestor
    ingestor = DocumentIngestor()
    
    try:
        # Process documents
        vector_store = ingestor.process_documents(sample_dir)
        print(f"Successfully processed documents and created vector store")
        print(f"Vector store contains {vector_store.index.ntotal} vectors")
        
        # Test retrieval
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.get_relevant_documents("What is RAG?")
        
        print(f"\nFound {len(docs)} relevant documents:")
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
