"""
FastAPI service for RAG application.
Provides REST API endpoints for document processing and question answering.
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
    """Request model for asking questions."""
    question: str = Field(..., description="The question to ask")
    top_k: Optional[int] = Field(4, description="Number of documents to retrieve")


class AnswerResponse(BaseModel):
    """Response model for answers."""
    answer: str = Field(..., description="The generated answer")
    question: str = Field(..., description="The original question")
    source_documents: List[Dict[str, Any]] = Field(..., description="Source documents used")
    num_sources: int = Field(..., description="Number of source documents")
    error: Optional[str] = Field(None, description="Error message if any")


class DocumentInfo(BaseModel):
    """Model for document information."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    rank: int = Field(..., description="Relevance rank")


class StatusResponse(BaseModel):
    """Response model for system status."""
    status: str = Field(..., description="System status")
    vector_store_loaded: bool = Field(..., description="Whether vector store is loaded")
    num_documents: Optional[int] = Field(None, description="Number of documents in vector store")


class HealthResponse(BaseModel):
    """Response model for health check."""
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
    Ask a question using the RAG system.
    
    Args:
        question: The question to ask
        top_k: Number of documents to retrieve for context
        
    Returns:
        Answer with source documents
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
    Upload documents to be processed by the RAG system.
    
    Args:
        files: List of uploaded files (.txt or .pdf)
        
    Returns:
        Status of document processing
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
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to enable auto-reload
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