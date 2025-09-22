# RAGStack — A Production-Ready RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system in Python for seamless document processing, intelligent querying, and multiple interaction interfaces. Perfect for building AI-powered document Q&A systems.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Option A: Streamlit Web UI (Recommended)](#option-a-streamlit-web-ui-recommended)
  - [Option B: FastAPI Service](#option-b-fastapi-service)
  - [Option C: Command Line](#option-c-command-line)
- [Detailed Usage](#detailed-usage)
  - [Document Ingestion](#document-ingestion)
  - [Question Answering](#question-answering)
  - [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
  - [Models](#models)
  - [Text Chunking Parameters](#text-chunking-parameters)
  - [Retrieval Parameters](#retrieval-parameters)
- [Example Use Cases](#example-use-cases)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Deployment](#deployment)
  - [Development](#development)
  - [Production (Docker)](#production-docker)
  - [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Support](#support)
- [Roadmap](#roadmap)

---

## Features

- 📄 **Document Ingestion**: `.txt` and `.pdf` with intelligent chunking  
- 🔍 **Vector Search**: FAISS-powered similarity search with Hugging Face embeddings  
- 🤖 **AI Question Answering**: Configurable LLMs via Hugging Face Transformers  
- 🌐 **REST API**: FastAPI service with comprehensive endpoints  
- 💻 **Web Interface**: Streamlit UI with document upload + chat  
- 📚 **Source Citations**: Transparent references for all answers  
- 🔧 **Modular Design**: Clean, production-ready code structure  

---

## Architecture

```text
RAG System Architecture

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Text Chunking  │───▶│   Embeddings    │
│  (.txt, .pdf)   │    │   (LangChain)    │    │ (Sentence-BERT) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│    Retrieval     │◀───│   Vector Store  │
│                 │    │   (Similarity)   │    │     (FAISS)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                        │
          │                        ▼
          │              ┌─────────────────┐
          └──────────────▶   LLM Answer    │
                         │  (Flan-T5/etc.) │
                         └─────────────────┘
```

---

## Project Structure

```text
.
├── ingest.py           # Document loading and vector store creation
├── qa.py               # RAG pipeline and question answering
├── api.py              # FastAPI REST service
├── app.py              # Streamlit web interface
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── vector_store/       # FAISS index storage (created automatically)
├── sample_docs/        # Sample documents (created automatically)
└── uploaded_docs/      # User-uploaded documents (created automatically)
```

---

## Quick Start

> **Prerequisites:** Python 3.9+

Clone the repository and install dependencies:

```bash
git clone <your-repo-url> RAGStack
cd RAGStack
pip install -r requirements.txt
```

### Option A: Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

Open: [http://localhost:8501](http://localhost:8501)

* Upload documents in the sidebar
* Ask questions in the main interface
* See answers with source citations

### Option B: FastAPI Service

Start the API server (either):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# or, if api.py includes a __main__ entrypoint:
python api.py
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option C: Command Line

```bash
# 1) Process documents
python ingest.py

# 2) Run Q&A
python qa.py
```

---

## Detailed Usage

### Document Ingestion

```python
from ingest import DocumentIngestor

ingestor = DocumentIngestor(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=100
)

vector_store = ingestor.process_documents("./my_documents")
```

### Question Answering

```python
from qa import RAGQuestionAnswerer

qa_system = RAGQuestionAnswerer(
    llm_model_name="google/flan-t5-base",
    top_k_retrieval=4
)

response = qa_system.ask_question("What is machine learning?")
print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['source_documents'])}")
```

### API Endpoints

* `GET  /` — API info
* `GET  /health` — Health check
* `GET  /status` — System status
* `GET  /ask?question=...` — Ask a question
* `POST /ask` — Ask a question (JSON payload)
* `GET  /search?query=...` — Search documents without answering
* `POST /upload-documents` — Upload new documents
* `GET  /conversation-history` — Get chat history
* `DELETE /conversation-history` — Clear chat history
* `PUT  /model?model_name=...` — Change LLM model

**Examples:**

```bash
# Ask a question
curl "http://localhost:8000/ask?question=What%20is%20RAG?"

# Upload documents
curl -X POST "http://localhost:8000/upload-documents" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"
```

---

## Configuration

### Models

**Embedding Models (document encoding):**

* `sentence-transformers/all-MiniLM-L6-v2` (default, fast)
* `sentence-transformers/all-mpnet-base-v2` (higher quality)
* `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (Q\&A optimized)

**LLM Models (answer generation):**

* `google/flan-t5-base` (default, balanced)
* `google/flan-t5-small` (faster, lower quality)
* `google/flan-t5-large` (slower, higher quality)
* `microsoft/DialoGPT-medium` (conversational)

### Text Chunking Parameters

```python
DocumentIngestor(
    chunk_size=500,        # Characters per chunk
    chunk_overlap=100,     # Overlap between chunks
    # ... other parameters
)
```

### Retrieval Parameters

```python
RAGQuestionAnswerer(
    top_k_retrieval=4,     # Documents to retrieve
    max_tokens=512,        # Max tokens in LLM response
    # ... other parameters
)
```

---

## Example Use Cases

* **Technical Documentation Q\&A** — Ask specific questions on your project docs
* **Research Paper Analysis** — Query for findings and references
* **Knowledge Base Search** — Company knowledge discovery
* **Educational Content** — AI tutor over textbooks/notes
* **Legal Document Review** — Contract searching and clause extraction

---

## Performance Tuning

**For Speed**

* Use smaller models (`flan-t5-small`, `all-MiniLM-L6-v2`)
* Reduce `chunk_size` and `top_k_retrieval`
* Prefer CPU-optimized settings

**For Quality**

* Larger models (`flan-t5-large`, `all-mpnet-base-v2`)
* Increase `top_k_retrieval` (6–8)
* Tune `chunk_size` per document type
* Use GPU acceleration if available

---

## Troubleshooting

**1) “No vector store found”**

```bash
# Process documents first
python ingest.py
# Or upload documents via the web UI
```

**2) Memory issues with large models**

```bash
# Use smaller models or adjust batch sizes
# For example, in qa.py set model to:
# "google/flan-t5-small"
```

**3) Slow performance**

```bash
# Reduce model size and retrieval count
# Use CPU-optimized dependencies in requirements.txt if needed
```

**4) PDF parsing issues**

```bash
# Ensure PDFs are text-based (not scanned)
pip install pdfplumber  # Alternate PDF parser
```

**Debugging**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check system status**

```bash
curl http://localhost:8000/status
```

---

## Advanced Features

**Custom Prompts (in `qa.py`)**

```python
prompt_template = """
Use the following context to answer the question.
Be concise and cite specific sources.

Context: {context}
Question: {question}
Answer: """
```

**Multiple Document Types (extend `ingest.py`)**

```python
def _load_docx_file(self, file_path: Path) -> List[Document]:
    """Load a Word document."""
    # TODO: implement .docx loader
    pass
```

**Custom Embedding Models**

```python
# Legal
embedding_model = "nlpaueb/legal-bert-base-uncased"

# Scientific
embedding_model = "allenai/scibert_scivocab_uncased"
```

**Batch Processing**

```python
folders = ["./docs1", "./docs2", "./docs3"]
for folder in folders:
    ingestor.process_documents(folder)
```

---

## API Reference

### `DocumentIngestor`

```python
class DocumentIngestor:
    def __init__(self, embedding_model, chunk_size, chunk_overlap, vector_store_path)
    def load_documents(self, folder_path) -> List[Document]
    def chunk_documents(self, documents) -> List[Document]
    def create_vector_store(self, documents) -> FAISS
    def save_vector_store(self) -> None
    def load_vector_store(self) -> Optional[FAISS]
    def process_documents(self, folder_path) -> FAISS
```

### `RAGQuestionAnswerer`

```python
class RAGQuestionAnswerer:
    def __init__(self, llm_model_name, vector_store_path, top_k_retrieval, max_tokens)
    def ask_question(self, question) -> Dict[str, Any]
    def update_model(self, new_model_name)
    def get_similar_documents(self, query, k) -> List[Dict[str, Any]]
    def reinitialize_with_new_documents(self, documents_path)
```

---

## Deployment

### Development

```bash
# Streamlit (port 8501)
streamlit run app.py

# FastAPI (port 8000)
uvicorn api:app --reload
```

### Production (Docker)

**Dockerfile**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & run:

```bash
docker build -t ragstack .
docker run -p 8000:8000 -p 8501:8501 ragstack
```

### Environment Variables

```bash
export MODEL_NAME=google/flan-t5-base
export VECTOR_STORE_PATH=./vector_store
export MAX_UPLOAD_SIZE=10MB
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

This project is open source under the MIT License.

---

## Acknowledgments

* **LangChain** — RAG framework
* **Hugging Face** — Transformers & embeddings
* **FAISS** — Efficient vector search
* **FastAPI** — Modern API framework
* **Streamlit** — Rapid UI development

---

## Support

* Check the **Troubleshooting** section
* Search existing GitHub issues
* Create a new issue with detailed information
* Join our community discussions

---

## Roadmap

* Support for more file formats (Word, Excel, etc.)
* Advanced filtering and metadata search
* Multi-language support
* Integration with cloud storage (S3, GCS)
* Fine-tuning for domain-specific models
* Conversation context awareness
* Performance monitoring & analytics
* Enterprise authentication & authorization

---

Happy questioning! 🤖📚

```
```
