"""
FastAPI service for RAG (Retrieval-Augmented Generation) application.

This module provides a RESTful API service for a question-answering system that combines
document retrieval with language model generation. It supports document upload, querying,
and conversation management through various endpoints.

Key Features:
    - Document upload and processing (.txt and .pdf)
    - Question answering with source citations
    - Document similarity search
    - Conversation history management
    - Model configuration
    - Health and status monitoring

The API uses FastAPI for high performance and automatic OpenAPI documentation generation.
All endpoints follow REST principles and include proper error handling.

Dependencies:
    - FastAPI: Web framework for building APIs
    - Uvicorn: ASGI server implementation
    - Pydantic: Data validation using Python type annotations
    - Custom RAG modules: qa.py and ingest.py for core functionality

Environment Variables:
    None required, but the following paths are used:
    - ./vector_store: For storing document embeddings
    - ./uploaded_docs: For temporary document storage

Example Usage:
    ```python
    # Run the server
    uvicorn api:app --host 0.0.0.0 --port 8000

    # In another process
    import requests
    response = requests.get("http://localhost:8000/ask?question=What is RAG?")
    print(response.json())
    ```
"""

import logging
import os
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from qa import RAGQuestionAnswerer, RAGChatBot
from ingest import DocumentIngestor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the RAG system
qa_system: Optional[RAGQuestionAnswerer] = None
chatbot: Optional[RAGChatBot] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    global qa_system, chatbot
    logger.info("Initializing RAG system...")
    
    try:
        qa_system = RAGQuestionAnswerer()
        chatbot = RAGChatBot(qa_system)
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        qa_system = None
        chatbot = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG system...")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Question-Answering API",
    description="A Retrieval-Augmented Generation API for document-based question answering",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QuestionRequest(BaseModel):
    """
    Pydantic model for question request body.

    This model validates the JSON payload for POST /ask endpoint, ensuring
    proper question format and optional retrieval parameters.

    Attributes:
        question: The user's question to be answered by the RAG system.
                 Must be a non-empty string.
        top_k: Number of relevant documents to retrieve for context.
               Defaults to 4, must be a positive integer.
               Larger values provide more context but may slow down response time.
    """
    question: str = Field(..., description="The question to ask")
    top_k: Optional[int] = Field(4, description="Number of documents to retrieve")


class AnswerResponse(BaseModel):
    """
    Pydantic model for question-answering response.

    This model structures the JSON response for both GET and POST /ask endpoints,
    including the generated answer, source documents, and any errors.

    Attributes:
        answer: The AI-generated answer based on retrieved documents.
        question: The original question for reference.
        source_documents: List of relevant document chunks used for the answer,
                        including content and metadata.
        num_sources: Total number of source documents used.
        error: Optional error message if something went wrong during processing.
               Null if successful.

    Example:
        {
            "answer": "RAG is a technique that combines...",
            "question": "What is RAG?",
            "source_documents": [{"content": "...", "metadata": {...}}],
            "num_sources": 2,
            "error": null
        }
    """
    answer: str = Field(..., description="The generated answer")
    question: str = Field(..., description="The original question")
    source_documents: List[Dict[str, Any]] = Field(..., description="Source documents used")
    num_sources: int = Field(..., description="Number of source documents")
    error: Optional[str] = Field(None, description="Error message if any")


class DocumentInfo(BaseModel):
    """
    Pydantic model for document chunk information.

    This model represents a single document chunk from the vector store,
    including its content, metadata, and relevance ranking. Used primarily
    by the /search endpoint.

    Attributes:
        content: The actual text content of the document chunk.
        metadata: Document metadata including file name, type, page numbers,
                 chunk indices, and other relevant information.
        rank: Relevance ranking (1-based) indicating how well this document
              matches the search query, with 1 being the most relevant.

    Example:
        {
            "content": "RAG (Retrieval-Augmented Generation) is...",
            "metadata": {
                "file_name": "intro.pdf",
                "page_number": 1,
                "chunk_index": 2
            },
            "rank": 1
        }
    """
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    rank: int = Field(..., description="Relevance rank")


class StatusResponse(BaseModel):
    """
    Pydantic model for system status information.

    This model provides the current state of the RAG system, including
    initialization status and document statistics. Used by the /status endpoint.

    Attributes:
        status: Current system state, one of:
               - "initialized": System is ready for use
               - "not_initialized": System is starting up or has failed
        vector_store_loaded: Boolean indicating whether the FAISS vector
                           store is loaded and ready for queries.
        num_documents: Optional count of document chunks in the vector store.
                      Null if vector store is not loaded.

    Example:
        {
            "status": "initialized",
            "vector_store_loaded": true,
            "num_documents": 150
        }
    """
    status: str = Field(..., description="System status")
    vector_store_loaded: bool = Field(..., description="Whether vector store is loaded")
    num_documents: Optional[int] = Field(None, description="Number of documents in vector store")


class HealthResponse(BaseModel):
    """
    Pydantic model for health check response.

    This model provides health status information about the service,
    used by the /health endpoint for monitoring and alerting.

    Attributes:
        status: Health status string, either:
               - "healthy": System is functioning normally
               - "unhealthy": System is experiencing issues
        message: Detailed message about the health status,
                useful for debugging and monitoring.

    Example:
        {
            "status": "healthy",
            "message": "RAG system is running"
        }
    """
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Health message")


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Question-Answering API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if qa_system is None:
        return HealthResponse(
            status="unhealthy",
            message="RAG system not initialized"
        )
    
    return HealthResponse(
        status="healthy",
        message="RAG system is running"
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status."""
    if qa_system is None:
        return StatusResponse(
            status="not_initialized",
            vector_store_loaded=False,
            num_documents=None
        )
    
    vector_store_loaded = qa_system.vector_store is not None
    num_documents = None
    
    if vector_store_loaded:
        try:
            num_documents = qa_system.vector_store.index.ntotal
        except:
            pass
    
    return StatusResponse(
        status="initialized",
        vector_store_loaded=vector_store_loaded,
        num_documents=num_documents
    )


@app.get("/ask", response_model=AnswerResponse)
async def ask_question(
    question: str = Query(..., description="The question to ask"),
    top_k: int = Query(4, description="Number of documents to retrieve", ge=1, le=10)
):
    """
    Ask a question and get an AI-generated answer with source citations.

    This endpoint processes a question using the RAG system, which:
    1. Retrieves relevant documents using semantic search
    2. Uses an LLM to generate an answer based on the retrieved context
    3. Returns the answer with source citations and document snippets

    Args:
        question: The question to ask. Should be a clear, well-formed question.
        top_k: Number of most relevant documents to retrieve (1-10).
              Higher values provide more context but may slow down the response.
              Default is 4, which balances accuracy and speed.

    Returns:
        AnswerResponse object containing:
        - Generated answer with source citations
        - Original question
        - Relevant document chunks with metadata
        - Number of sources used
        - Error message if any

    Raises:
        HTTPException(503): If RAG system is not initialized
        HTTPException(400): If question is empty
        HTTPException(500): For processing errors

    Example Request:
        GET /ask?question=What%20is%20RAG%3F&top_k=4

    Example Response:
        {
            "answer": "RAG (Retrieval-Augmented Generation) is a technique that [intro.txt p1]...",
            "question": "What is RAG?",
            "source_documents": [...],
            "num_sources": 4,
            "error": null
        }
    """
    if qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized"
        )
    
    if not question.strip():
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty"
        )
    
    try:
        # Temporarily update top_k if different from default
        original_k = qa_system.top_k_retrieval
        if top_k != original_k:
            qa_system.top_k_retrieval = top_k
            qa_system._setup_qa_chain()
        
        # Get answer
        result = qa_system.ask_question(question)
        
        # Restore original top_k
        if top_k != original_k:
            qa_system.top_k_retrieval = original_k
            qa_system._setup_qa_chain()
        
        return AnswerResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question_post(request: QuestionRequest):
    """
    Ask a question using POST method.
    
    Args:
        request: Question request with question and optional top_k
        
    Returns:
        Answer with source documents
    """
    if qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized"
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty"
        )
    
    try:
        # Temporarily update top_k if different from default
        original_k = qa_system.top_k_retrieval
        if request.top_k != original_k:
            qa_system.top_k_retrieval = request.top_k
            qa_system._setup_qa_chain()
        
        # Get answer
        result = qa_system.ask_question(request.question)
        
        # Restore original top_k
        if request.top_k != original_k:
            qa_system.top_k_retrieval = original_k
            qa_system._setup_qa_chain()
        
        return AnswerResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/search", response_model=List[DocumentInfo])
async def search_documents(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of documents to return", ge=1, le=20)
):
    """
    Search for similar documents without generating an answer.
    
    Args:
        query: The search query
        k: Number of documents to return
        
    Returns:
        List of similar documents
    """
    if qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized"
        )
    
    if not query.strip():
        raise HTTPException(
            status_code=400, 
            detail="Query cannot be empty"
        )
    
    try:
        documents = qa_system.get_similar_documents(query, k)
        return [DocumentInfo(**doc) for doc in documents]
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )


@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents to be used by the RAG system.

    This endpoint handles file uploads and document processing by:
    1. Validating file types (.txt and .pdf only)
    2. Saving files to a temporary directory
    3. Processing documents into chunks
    4. Creating or updating the vector store
    5. Cleaning up temporary files

    Args:
        files: List of files to upload. Each file must be either:
              - A text file (.txt)
              - A PDF document (.pdf)
              Files should be text-based (not scanned images).

    Returns:
        Dict containing:
        - Success/error message
        - List of processed files
        - Processing status

    Raises:
        HTTPException(400): If no files provided or invalid file type
        HTTPException(500): For processing errors

    Example Response:
        {
            "message": "Successfully uploaded and processed 2 files",
            "files": ["document1.pdf", "document2.txt"],
            "status": "success"
        }

    Notes:
        - Large files may take longer to process
        - PDF files must be text-based for proper extraction
        - Previous vector store will be updated with new documents
        - Processing includes text extraction, chunking, and embedding
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    # Create upload directory
    upload_dir = "./uploaded_docs"
    os.makedirs(upload_dir, exist_ok=True)
    
    uploaded_files = []
    
    try:
        # Save uploaded files
        for file in files:
            if not file.filename:
                continue
                
            # Check file type
            file_ext = file.filename.lower().split('.')[-1]
            if file_ext not in ['txt', 'pdf']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Only .txt and .pdf files are supported."
                )
            
            # Save file
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append(file.filename)
        
        # Reinitialize RAG system with new documents
        if qa_system:
            qa_system.reinitialize_with_new_documents(upload_dir)
        
        return {
            "message": f"Successfully uploaded and processed {len(uploaded_files)} files",
            "files": uploaded_files,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading documents: {str(e)}"
        )


@app.get("/conversation-history")
async def get_conversation_history():
    """
    Get the conversation history from the chatbot.
    
    Returns:
        List of conversation entries
    """
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized"
        )
    
    return {
        "history": chatbot.get_conversation_history(),
        "count": len(chatbot.get_conversation_history())
    }


@app.delete("/conversation-history")
async def clear_conversation_history():
    """
    Clear the conversation history.
    
    Returns:
        Confirmation message
    """
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized"
        )
    
    chatbot.clear_history()
    
    return {
        "message": "Conversation history cleared",
        "status": "success"
    }


@app.put("/model")
async def update_model(model_name: str = Query(..., description="New model name")):
    """
    Update the LLM model used by the RAG system.
    
    Args:
        model_name: Name of the new Hugging Face model
        
    Returns:
        Status of model update
    """
    if qa_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    if not model_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Model name cannot be empty"
        )
    
    try:
        qa_system.update_model(model_name)
        
        return {
            "message": f"Successfully updated model to {model_name}",
            "model": model_name,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating model: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {"error": "Endpoint not found", "status_code": 404}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return {"error": "Internal server error", "status_code": 500}


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """
    Run the FastAPI server with the specified configuration.

    This function starts the Uvicorn ASGI server with the FastAPI application.
    It's used when running the API directly (not through an external ASGI server).

    Args:
        host: Network interface to bind to:
              - "127.0.0.1" (default) for local access only
              - "0.0.0.0" for all network interfaces
        port: TCP port to listen on (default: 8000)
        reload: Whether to enable auto-reload on code changes.
               Useful during development, should be False in production.

    Example Usage:
        ```python
        # Run locally for development
        run_server()

        # Run on all interfaces for production
        run_server(host="0.0.0.0", port=80, reload=False)
        ```

    Notes:
        - In production, consider using a proper ASGI server configuration
        - The server provides automatic API documentation at /docs
        - For HTTPS, configure the ASGI server with SSL certificates
    """
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()