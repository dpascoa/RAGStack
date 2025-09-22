RAG Question-Answering System

A comprehensive Retrieval-Augmented Generation (RAG) application built with Python, featuring document ingestion, vector search, and AI-powered question answering with both API and web interfaces.
Features

    📄 Document Ingestion: Support for .txt and .pdf files with intelligent chunking
    🔍 Vector Search: FAISS-powered similarity search with Hugging Face embeddings
    🤖 AI Question Answering: Configurable LLM models via Hugging Face Transformers
    🌐 REST API: FastAPI-based service with comprehensive endpoints
    💻 Web Interface: Interactive Streamlit UI with document upload and chat
    📚 Source Citations: Transparent source document references for all answers
    🔧 Modular Design: Clean, production-ready code structure

Architecture

RAG System Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Text Chunking  │───▶│   Embeddings    │
│  (.txt, .pdf)   │    │   (LangChain)    │    │ (Sentence-BERT) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Retrieval      │◀───│  Vector Store   │
│                 │    │   (Similarity)   │    │    (FAISS)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                        │
          │                        ▼
          │              ┌─────────────────┐
          └──────────────▶│   LLM Answer    │
                         │ (Flan-T5/etc.)  │
                         └─────────────────┘

Project Structure

rag-application/
├── ingest.py           # Document loading and vector store creation
├── qa.py              # RAG pipeline and question answering
├── api.py             # FastAPI REST service
├── app.py             # Streamlit web interface
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── vector_store/      # FAISS index storage (created automatically)
├── sample_docs/       # Sample documents (created automatically)
└── uploaded_docs/     # User-uploaded documents (created automatically)

Quick Start
1. Installation
bash

# Clone the repository (if you haven't already)
mkdir rag-application
cd rag-application

# Install dependencies
pip install -r requirements.txt

2. Basic Usage
Option A: Streamlit Web Interface (Recommended)
bash

# Start the web interface
streamlit run app.py

Then open your browser to http://localhost:8501 and:

    Upload documents using the sidebar
    Ask questions in the main interface
    View answers with source citations

Option B: FastAPI Service
bash

# Start the API server
python api.py

Then open http://localhost:8000/docs for the interactive API documentation.
Option C: Command Line
bash

# First, process some documents
python ingest.py

# Then run Q&A
python qa.py

Detailed Usage Guide
Document Ingestion

The system automatically handles document processing:
python

from ingest import DocumentIngestor

# Initialize ingestor
ingestor = DocumentIngestor(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=100
)

# Process documents from a folder
vector_store = ingestor.process_documents("./my_documents")

Question Answering
python

from qa import RAGQuestionAnswerer

# Initialize Q&A system
qa_system = RAGQuestionAnswerer(
    llm_model_name="google/flan-t5-base",
    top_k_retrieval=4
)

# Ask a question
response = qa_system.ask_question("What is machine learning?")
print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['source_documents'])}")

API Endpoints

The FastAPI service provides these key endpoints:

    GET /ask?question=... - Ask a question
    POST /ask - Ask a question (JSON payload)
    GET /search?query=... - Search documents without answering
    POST /upload-documents - Upload new documents
    GET /status - System status
    PUT /model?model_name=... - Change LLM model

Example API usage:
bash

# Ask a question
curl "http://localhost:8000/ask?question=What%20is%20RAG?"

# Upload documents
curl -X POST "http://localhost:8000/upload-documents" \
     -F "files=@document1.pdf" \
     -F "files=@document2.txt"

Configuration Options
Models

Embedding Models (for document encoding):

    sentence-transformers/all-MiniLM-L6-v2 (default, fast)
    sentence-transformers/all-mpnet-base-v2 (higher quality)
    sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (optimized for Q&A)

LLM Models (for answer generation):

    google/flan-t5-base (default, balanced)
    google/flan-t5-small (faster, lower quality)
    google/flan-t5-large (slower, higher quality)
    microsoft/DialoGPT-medium (conversational)

Text Chunking Parameters
python

DocumentIngestor(
    chunk_size=500,        # Characters per chunk
    chunk_overlap=100,     # Overlap between chunks
    # ... other parameters
)

Retrieval Parameters
python

RAGQuestionAnswerer(
    top_k_retrieval=4,     # Documents to retrieve
    max_tokens=512,        # Max tokens in LLM response
    # ... other parameters
)

Example Use Cases

    Technical Documentation Q&A: Upload your project docs and ask specific technical questions
    Research Paper Analysis: Process research papers and query for specific findings
    Knowledge Base Search: Create a searchable knowledge base from company documents
    Educational Content: Upload textbooks and create an AI tutor
    Legal Document Review: Search through contracts and legal documents

Performance Optimization
For Better Speed:

    Use smaller models (flan-t5-small, all-MiniLM-L6-v2)
    Reduce chunk_size and top_k_retrieval
    Use CPU-optimized settings

For Better Quality:

    Use larger models (flan-t5-large, all-mpnet-base-v2)
    Increase top_k_retrieval (6-8 documents)
    Fine-tune chunk_size based on document type
    Use GPU acceleration if available

Troubleshooting
Common Issues

1. "No vector store found" error:
bash

# Solution: Process documents first
python ingest.py
# Or upload documents via the web interface

2. Memory issues with large models:
bash

# Solution: Use smaller models or adjust batch sizes
# Edit qa.py and change model to "google/flan-t5-small"

3. Slow performance:
bash

# Solution: Reduce model size and retrieval count
# Use CPU-optimized settings in requirements.txt

4. PDF parsing issues:
bash

# Solution: Ensure PDF files are text-based (not scanned images)
# Install additional dependencies if needed:
pip install pdfplumber  # Alternative PDF parser

Debugging

Enable detailed logging:
python

import logging
logging.basicConfig(level=logging.DEBUG)

Check system status via API:
bash

curl http://localhost:8000/status

Advanced Features
Custom Prompts

Modify the prompt template in qa.py:
python

prompt_template = """
Use the following context to answer the question.
Be concise and cite specific sources.

Context: {context}
Question: {question}
Answer: """

Multiple Document Types

Extend ingest.py to support more file types:
python

def _load_docx_file(self, file_path: Path) -> List[Document]:
    """Load a Word document."""
    # Implementation for .docx files
    pass

Custom Embedding Models

Use domain-specific embedding models:
python

# For legal documents
embedding_model = "nlpaueb/legal-bert-base-uncased"

# For scientific papers
embedding_model = "allenai/scibert_scivocab_uncased"

Batch Processing

Process multiple document folders:
python

folders = ["./docs1", "./docs2", "./docs3"]
for folder in folders:
    ingestor.process_documents(folder)

API Reference
DocumentIngestor
python

class DocumentIngestor:
    def __init__(self, embedding_model, chunk_size, chunk_overlap, vector_store_path)
    def load_documents(self, folder_path) -> List[Document]
    def chunk_documents(self, documents) -> List[Document]
    def create_vector_store(self, documents) -> FAISS
    def save_vector_store(self) -> None
    def load_vector_store(self) -> Optional[FAISS]
    def process_documents(self, folder_path) -> FAISS

RAGQuestionAnswerer
python

class RAGQuestionAnswerer:
    def __init__(self, llm_model_name, vector_store_path, top_k_retrieval, max_tokens)
    def ask_question(self, question) -> Dict[str, Any]
    def update_model(self, new_model_name)
    def get_similar_documents(self, query, k) -> List[Dict[str, Any]]
    def reinitialize_with_new_documents(self, documents_path)

FastAPI Endpoints

GET  /                      # API information
GET  /health                # Health check
GET  /status                # System status
GET  /ask                   # Ask question (query param)
POST /ask                   # Ask question (JSON body)
GET  /search                # Search documents
POST /upload-documents      # Upload files
GET  /conversation-history  # Get chat history
DELETE /conversation-history # Clear chat history
PUT  /model                 # Update LLM model

Deployment
Development
bash

# Streamlit (port 8501)
streamlit run app.py

# FastAPI (port 8000)
uvicorn api:app --reload

Production

Docker Deployment:

Create Dockerfile:
dockerfile

FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

Environment Variables:
bash

export MODEL_NAME=google/flan-t5-base
export VECTOR_STORE_PATH=./vector_store
export MAX_UPLOAD_SIZE=10MB

Contributing

    Fork the repository
    Create a feature branch (git checkout -b feature/amazing-feature)
    Commit changes (git commit -m 'Add amazing feature')
    Push to branch (git push origin feature/amazing-feature)
    Open a Pull Request

License

This project is open source and available under the MIT License.
Acknowledgments

    LangChain for RAG framework
    Hugging Face for transformers and embeddings
    FAISS for efficient vector search
    FastAPI for modern API framework
    Streamlit for rapid UI development

Support

For issues and questions:

    Check the Troubleshooting section
    Search existing GitHub issues
    Create a new issue with detailed information
    Join our community discussions

Roadmap

    Support for more file formats (Word, Excel, etc.)
    Advanced filtering and metadata search
    Multi-language support
    Integration with cloud storage (S3, GCS)
    Fine-tuning capabilities for domain-specific models
    Conversation context awareness
    Performance monitoring and analytics
    Enterprise authentication and authorization

Happy questioning! 🤖📚
