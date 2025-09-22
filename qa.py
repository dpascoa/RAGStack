"""
Question-Answering module for RAG application.
Implements the RAG pipeline using LangChain.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import os

from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from ingest import DocumentIngestor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGQuestionAnswerer:
    """
    Handles question-answering using Retrieval-Augmented Generation.
    """
    
    def __init__(
        self,
        llm_model_name: str = "google/flan-t5-base",
        vector_store_path: str = "./vector_store",
        top_k_retrieval: int = 4,
        max_tokens: int = 512
    ):
        """
        Initialize the RAG Q&A system.
        
        Args:
            llm_model_name: Name of the Hugging Face LLM model
            vector_store_path: Path to the vector store
            top_k_retrieval: Number of documents to retrieve
            max_tokens: Maximum tokens for LLM generation
        """
        self.llm_model_name = llm_model_name
        self.vector_store_path = vector_store_path
        self.top_k_retrieval = top_k_retrieval
        self.max_tokens = max_tokens
        
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        self.ingestor = DocumentIngestor(vector_store_path=vector_store_path)
        
        # Initialize components
        self._setup_llm()
        self._load_vector_store()
        self._setup_qa_chain()
    
    def _setup_llm(self):
        """Initialize the LLM pipeline."""
        logger.info(f"Loading LLM model: {self.llm_model_name}")
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.llm_model_name,
                dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Create pipeline
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=self.max_tokens,
                temperature=0.1,
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Wrap in LangChain
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM: {str(e)}")
            raise
    
    def _load_vector_store(self):
        """Load the vector store."""
        logger.info("Loading vector store...")
        
        self.vector_store = self.ingestor.load_vector_store()
        
        if self.vector_store is None:
            logger.warning("No vector store found. Please ingest documents first.")
        else:
            logger.info("Vector store loaded successfully")
    
    def _setup_qa_chain(self):
        """Setup the QA chain with custom prompt."""
        if self.vector_store is None:
            logger.warning("Cannot setup QA chain without vector store")
            return
        
        # Custom prompt template
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, 
        don't try to make up an answer.

        Context: {context}

        Question: {question}
        
        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k_retrieval}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        logger.info("QA chain setup complete")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with source documents.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            return {
                "error": "QA chain not initialized. Please ensure vector store is loaded.",
                "answer": "",
                "source_documents": []
            }
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Get answer from QA chain
            result = self.qa_chain.invoke({"query": question})
            
            # Extract answer and sources
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Format source documents
            formatted_sources = []
            for i, doc in enumerate(source_docs):
                source_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_rank": i + 1
                }
                formatted_sources.append(source_info)
            
            response = {
                "answer": answer.strip(),
                "source_documents": formatted_sources,
                "question": question,
                "num_sources": len(formatted_sources)
            }
            
            logger.info(f"Question answered successfully with {len(formatted_sources)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "error": f"Error processing question: {str(e)}",
                "answer": "",
                "source_documents": [],
                "question": question
            }
    
    def update_model(self, new_model_name: str):
        """
        Update the LLM model.
        
        Args:
            new_model_name: Name of the new model to use
        """
        logger.info(f"Updating model from {self.llm_model_name} to {new_model_name}")
        
        self.llm_model_name = new_model_name
        self._setup_llm()
        self._setup_qa_chain()
        
        logger.info("Model updated successfully")
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar documents for a query without generating an answer.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents with metadata
        """
        if self.vector_store is None:
            return []
        
        try:
            # Get similar documents
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": k}
            )
            docs = retriever.get_relevant_documents(query)
            
            # Format documents
            formatted_docs = []
            for i, doc in enumerate(docs):
                doc_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "rank": i + 1
                }
                formatted_docs.append(doc_info)
            
            return formatted_docs
            
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {str(e)}")
            return []
    
    def reinitialize_with_new_documents(self, documents_path: str):
        """
        Reinitialize the system with new documents.
        
        Args:
            documents_path: Path to folder containing new documents
        """
        logger.info("Reinitializing with new documents...")
        
        # Process new documents
        self.ingestor.process_documents(documents_path)
        
        # Reload vector store and setup QA chain
        self._load_vector_store()
        self._setup_qa_chain()
        
        logger.info("Reinitialization complete")


class RAGChatBot:
    """
    A simple chatbot interface for the RAG system with conversation history.
    """
    
    def __init__(self, qa_system: RAGQuestionAnswerer):
        """
        Initialize the chatbot.
        
        Args:
            qa_system: The RAG question answerer system
        """
        self.qa_system = qa_system
        self.conversation_history = []
    
    def chat(self, message: str) -> Dict[str, Any]:
        """
        Process a chat message and return response.
        
        Args:
            message: User message
            
        Returns:
            Response dictionary with answer and context
        """
        # Get answer
        response = self.qa_system.ask_question(message)
        
        # Add to conversation history
        conversation_entry = {
            "question": message,
            "answer": response.get("answer", ""),
            "sources": response.get("source_documents", []),
            "timestamp": None  # Could add timestamp if needed
        }
        self.conversation_history.append(conversation_entry)
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []


def main():
    """Example usage of the RAG Q&A system."""
    # Initialize the Q&A system
    qa_system = RAGQuestionAnswerer()
    
    if qa_system.vector_store is None:
        print("No vector store found. Please run ingest.py first to process documents.")
        return
    
    # Initialize chatbot
    chatbot = RAGChatBot(qa_system)
    
    # Example questions
    example_questions = [
        "What is RAG?",
        "How do vector databases work?",
        "What is FAISS used for?",
    ]
    
    print("RAG Question-Answering System")
    print("=" * 40)
    
    for question in example_questions:
        print(f"\nQuestion: {question}")
        response = chatbot.chat(question)
        
        if "error" in response:
            print(f"Error: {response['error']}")
        else:
            print(f"Answer: {response['answer']}")
            print(f"Sources used: {response['num_sources']}")
            
            # Show first source
            if response['source_documents']:
                first_source = response['source_documents'][0]
                print(f"First source: {first_source['content'][:150]}...")
    
    print(f"\nConversation history has {len(chatbot.get_conversation_history())} entries")


if __name__ == "__main__":
    main()