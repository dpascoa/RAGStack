"""
Document ingestion module for RAG application.
Handles loading, chunking, and embedding of documents.
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
    Handles document ingestion, chunking, and vector store creation.
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
        Load documents from a folder, supporting .txt and .pdf files.
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            List of loaded documents
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
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
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
        Create a FAISS vector store from documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            FAISS vector store
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
        Complete document processing pipeline.
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            FAISS vector store
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
