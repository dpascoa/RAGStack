"""
Question-Answering module for RAG application.
Implements the RAG pipeline using LangChain.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import os

from langchain_huggingface import HuggingFacePipeline
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
        self.retriever = None
        self.answer_prompt: Optional[PromptTemplate] = None
        self.document_prompt: Optional[PromptTemplate] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        # Reserve tokens for the question and instructions to stay within model limits
        self.max_context_tokens = 360
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
            
            # Create deterministic generation pipeline for grounded answers
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=0.0,
                do_sample=False,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Wrap in LangChain
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.tokenizer = tokenizer
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
        
        # Custom prompt template encouraging grounded, cited answers
        prompt_template = """
        You are a meticulous assistant that answers using only the provided context.
        Follow the rules:
        - If the context does not contain the answer, reply "I don't know based on the provided documents."
        - For every statement you make, cite the supporting source between square brackets.
        - Prefer the format [file_name pX] when a page_number exists, otherwise use [file_name chunk chunk_index].
        - Quote the most relevant snippet before adding any explanation when helpful.

        Context:
        {context}

        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        document_prompt = PromptTemplate(
            template=(
                "Source: {file_name} (page {page_number}) | chunk {chunk_index}/{total_chunks}\n"
                "{page_content}\n"
            ),
            input_variables=[
                "page_content",
                "file_name",
                "page_number",
                "chunk_index",
                "total_chunks"
            ]
        )

        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.top_k_retrieval,
                "fetch_k": max(self.top_k_retrieval * 2, 8),
                "lambda_mult": 0.6
            }
        )

        self.answer_prompt = PROMPT
        self.document_prompt = document_prompt

        logger.info("QA chain setup complete")

    def _prepare_context(self, documents: List[Document]) -> Tuple[str, List[Document]]:
        """Select and format documents so the prompt stays within token limits."""
        if not documents:
            return "", []

        selected_docs: List[Document] = []
        context_segments: List[str] = []
        tokens_used = 0

        for doc in documents:
            formatted = self._format_document_for_context(doc)
            if not formatted.strip():
                continue

            doc_tokens = self._count_tokens(formatted)
            if doc_tokens == 0:
                continue

            if tokens_used + doc_tokens > self.max_context_tokens:
                remaining = max(self.max_context_tokens - tokens_used, 0)
                if remaining <= 0:
                    break
                trimmed = self._trim_to_token_limit(formatted, remaining)
                if trimmed.strip():
                    context_segments.append(trimmed)
                    selected_docs.append(doc)
                break

            context_segments.append(formatted)
            selected_docs.append(doc)
            tokens_used += doc_tokens

            if tokens_used >= self.max_context_tokens:
                break

        context_text = "\n\n".join(context_segments)
        return context_text, selected_docs

    def _format_document_for_context(self, doc: Document) -> str:
        """Format a document chunk with metadata for inclusion in the prompt."""
        if self.document_prompt is None:
            return doc.page_content

        values = {
            "page_content": doc.page_content,
            "file_name": str(doc.metadata.get("file_name", "unknown")),
            "page_number": str(doc.metadata.get("page_number", "n/a")),
            "chunk_index": str(doc.metadata.get("chunk_index", "n/a")),
            "total_chunks": str(doc.metadata.get("total_chunks", "n/a"))
        }
        return self.document_prompt.format(**values)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model tokenizer, fallback to char heuristic."""
        if not text:
            return 0
        if self.tokenizer is None:
            return max(len(text) // 4, 1)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _trim_to_token_limit(self, text: str, token_limit: int) -> str:
        """Trim text to stay within the provided token limit."""
        if token_limit <= 0 or not text:
            return ""

        if self.tokenizer is None:
            # Roughly approximate with characters if tokenizer is unavailable
            approx_chars = token_limit * 4
            return text[:approx_chars]

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= token_limit:
            return text

        trimmed_ids = token_ids[:token_limit]
        return self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with source documents.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.retriever is None or self.answer_prompt is None:
            return {
                "error": "QA chain not initialized. Please ensure vector store is loaded.",
                "answer": "",
                "source_documents": []
            }

        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            retrieved_docs = self.retriever.get_relevant_documents(question)
            context, used_docs = self._prepare_context(retrieved_docs)

            if not context.strip():
                context = "No relevant supporting context was retrieved."

            prompt = self.answer_prompt.format(context=context, question=question)
            answer = self.llm.invoke(prompt)

            if isinstance(answer, str):
                answer_text = answer.strip()
            else:
                answer_text = str(answer)

            # Format source documents
            formatted_sources = []
            for i, doc in enumerate(used_docs):
                source_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_rank": i + 1
                }
                formatted_sources.append(source_info)

            response = {
                "answer": answer_text,
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
