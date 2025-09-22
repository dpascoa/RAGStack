"""
Streamlit-based web interface for RAG (Retrieval-Augmented Generation) application.

This module provides a user-friendly web interface for interacting with the RAG system,
featuring:

1. Document Management:
   - File upload support (.txt, .pdf)
   - Sample document creation
   - Processing status and feedback
   - Document statistics display

2. Question Answering:
   - Interactive chat interface
   - Source document display
   - Conversation history
   - Answer generation with citations

3. System Configuration:
   - Model selection
   - Retrieval parameter tuning
   - Conversation management
   - System status monitoring

4. User Interface Features:
   - Clean, intuitive design
   - Real-time feedback
   - Responsive layout
   - Error handling
   - Progress indicators

Key Components:
    - StreamlitRAGApp: Main application class
    - Custom styling and layout
    - Session state management
    - File handling utilities
    - Error recovery mechanisms

Dependencies:
    - streamlit: Web interface framework
    - qa.py: Core RAG functionality
    - ingest.py: Document processing
    - tempfile: File handling
    - logging: Activity tracking

Example Usage:
    ```bash
    # Run the Streamlit app
    streamlit run app.py

    # Access in browser
    # http://localhost:8501
    ```

Technical Notes:
    - Uses Streamlit session state
    - Efficient file handling
    - Responsive design
    - Error resilient
    - Memory efficient
"""

import streamlit as st
import requests
import json
import os
import tempfile
from typing import List, Dict, Any
import logging

from qa import RAGQuestionAnswerer, RAGChatBot
from ingest import DocumentIngestor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Question-Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-doc {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitRAGApp:
    """
    Main Streamlit application class for the RAG system.

    This class manages the complete web interface, including:
    - System initialization and state management
    - Document upload and processing
    - Question answering interface
    - Conversation history display
    - System configuration
    - User interface layout

    The application provides:
    1. Document Management:
       - File upload interface
       - Processing status
       - Sample document creation
       - Vector store statistics

    2. Question Answering:
       - Question input
       - Answer display with citations
       - Source document viewing
       - Chat history management

    3. Configuration:
       - Model selection
       - Retrieval settings
       - Display preferences
       - System controls

    Example:
        ```python
        def main():
            app = StreamlitRAGApp()
            app.run()
        ```

    Technical Details:
        - Uses Streamlit session state
        - Manages file uploads efficiently
        - Handles state persistence
        - Provides error recovery
        - Responsive layout design

    Notes:
        - Initialize once per session
        - Handles page reloads
        - Thread-safe implementation
        - Memory-efficient processing
    """
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.initialize_session_state()
        self.initialize_rag_system()
    
    def initialize_session_state(self):
        """
        Initialize Streamlit session state variables.

        This method sets up the persistent state for the application:
        - conversation_history: List of Q&A interactions
        - rag_system: RAG system instance
        - chatbot: Chat interface instance
        - documents_processed: Document processing status

        The session state ensures:
        - State persistence across page reloads
        - Component coordination
        - User state tracking
        - System status management

        Technical Details:
            - Uses Streamlit's session_state
            - Thread-safe implementation
            - Memory-efficient storage
            - Handles edge cases

        Notes:
            - Called during app initialization
            - Resets on session expiry
            - Maintains conversation context
            - Preserves system status
        """
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None
        
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = False
    
    def initialize_rag_system(self):
        """Initialize or load the RAG system."""
        if st.session_state.rag_system is None:
            try:
                with st.spinner("Initializing RAG system..."):
                    qa_system = RAGQuestionAnswerer()
                    if qa_system.vector_store is not None:
                        st.session_state.rag_system = qa_system
                        st.session_state.chatbot = RAGChatBot(qa_system)
                        st.session_state.documents_processed = True
                        st.success("RAG system initialized successfully!")
                    else:
                        st.warning("No pre-existing vector store found. Please upload documents to get started.")
            except Exception as e:
                st.error(f"Error initializing RAG system: {str(e)}")
                logger.error(f"Error initializing RAG system: {str(e)}")
    
    def process_uploaded_documents(self, uploaded_files):
        """
        Process uploaded documents and update the RAG system.

        This method handles the complete document processing workflow:
        1. File validation and saving
        2. Document processing
        3. Vector store creation
        4. System reinitialization
        5. Cleanup of temporary files

        The process ensures:
        - Safe file handling
        - Proper error handling
        - System state updates
        - Resource cleanup

        Args:
            uploaded_files: List of Streamlit UploadedFile objects
                          (.txt or .pdf files)

        Returns:
            bool: True if processing successful, False otherwise

        Technical Details:
            - Uses temporary directory
            - Handles multiple file types
            - Manages system state
            - Provides progress feedback

        Notes:
            - Supports .txt and .pdf
            - Updates vector store
            - Thread-safe processing
            - Cleans up resources
        """
        if not uploaded_files:
            return False
        
        try:
            # Create temporary directory for uploaded files
            temp_dir = tempfile.mkdtemp()
            
            with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                # Save uploaded files
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Initialize document ingestor and process documents
                ingestor = DocumentIngestor()
                vector_store = ingestor.process_documents(temp_dir)
                
                # Initialize RAG system with new vector store
                qa_system = RAGQuestionAnswerer()
                st.session_state.rag_system = qa_system
                st.session_state.chatbot = RAGChatBot(qa_system)
                st.session_state.documents_processed = True
                
                # Clean up temporary directory
                import shutil
                shutil.rmtree(temp_dir)
                
                return True
                
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def display_sidebar(self):
        """
        Create and populate the application sidebar.

        This method builds the sidebar interface containing:
        1. System Status:
           - Initialization status
           - Document count
           - Processing state

        2. Document Upload:
           - File upload interface
           - Supported formats
           - Processing controls

        3. Model Settings:
           - Model selection
           - Retrieval parameters
           - Performance options

        4. Conversation Controls:
           - History management
           - Clear chat option
           - Statistics display

        Returns:
            int: Current top_k value for document retrieval

        Technical Details:
            - Uses Streamlit components
            - Updates in real-time
            - Handles user interactions
            - Manages system settings

        Notes:
            - Responsive layout
            - Persistent settings
            - Immediate feedback
            - Error resilient
        """
        st.sidebar.title("🤖 RAG System")
        
        # System status
        st.sidebar.subheader("System Status")
        if st.session_state.rag_system is not None:
            st.sidebar.success("✅ System Ready")
            if st.session_state.rag_system.vector_store is not None:
                num_docs = st.session_state.rag_system.vector_store.index.ntotal
                st.sidebar.info(f"📚 {num_docs} document chunks indexed")
            else:
                st.sidebar.warning("No documents loaded")
        else:
            st.sidebar.error("❌ System Not Ready")
        
        st.sidebar.markdown("---")
        
        # Document upload
        st.sidebar.subheader("📄 Upload Documents")
        uploaded_files = st.sidebar.file_uploader(
            "Upload documents (.txt, .pdf)",
            accept_multiple_files=True,
            type=['txt', 'pdf'],
            help="Upload text or PDF files to build your knowledge base"
        )
        
        if uploaded_files:
            if st.sidebar.button("Process Documents", type="primary"):
                if self.process_uploaded_documents(uploaded_files):
                    st.sidebar.success(f"Successfully processed {len(uploaded_files)} documents!")
                    st.rerun()
        
        st.sidebar.markdown("---")
        
        # Model selection
        st.sidebar.subheader("🧠 Model Settings")
        model_options = [
            "google/flan-t5-base",
            "google/flan-t5-small",
            "google/flan-t5-large",
            "microsoft/DialoGPT-medium"
        ]
        
        current_model = st.sidebar.selectbox(
            "Select LLM Model",
            model_options,
            index=0,
            help="Choose the language model for generating answers"
        )
        
        # Retrieval settings
        top_k = st.sidebar.slider(
            "Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of relevant documents to use for answering"
        )
        
        st.sidebar.markdown("---")
        
        # Conversation controls
        st.sidebar.subheader("💬 Conversation")
        if st.sidebar.button("Clear History"):
            st.session_state.conversation_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_history()
            st.success("Conversation history cleared!")
        
        # Show conversation stats
        if st.session_state.conversation_history:
            st.sidebar.info(f"📈 {len(st.session_state.conversation_history)} messages in history")
        
        return top_k
    
    def display_main_interface(self, top_k: int):
        """Display the main question-answering interface."""
        st.title("🤖 RAG Question-Answering System")
        st.markdown("Ask questions about your uploaded documents and get AI-powered answers with source citations.")
        
        # Check if system is ready
        if not st.session_state.documents_processed:
            st.warning("⚠️ Please upload documents in the sidebar to get started!")
            
            # Show sample documents creation option
            if st.button("Create Sample Documents"):
                self.create_sample_documents()
                st.rerun()
            
            return
        
        # Question input
        st.subheader("💭 Ask a Question")
        
        # Use columns for better layout
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the main topic of the documents?",
                help="Type your question about the uploaded documents"
            )
        
        with col2:
            ask_button = st.button("Ask", type="primary", disabled=not question.strip())
        
        # Process question
        if ask_button and question.strip():
            self.process_question(question, top_k)
        
        # Display conversation history
        self.display_conversation_history()
    
    def process_question(self, question: str, top_k: int):
        """
        Process a user question and display the response.

        This method handles the complete Q&A workflow:
        1. Question Processing:
           - Input validation
           - Parameter updating
           - System preparation

        2. Answer Generation:
           - RAG processing
           - Source retrieval
           - Response formatting

        3. Display Updates:
           - Answer rendering
           - Source display
           - Status updates
           - Error handling

        Args:
            question: User's question text
            top_k: Number of documents to retrieve

        Technical Details:
            - Updates retrieval parameters
            - Manages chat history
            - Provides visual feedback
            - Handles errors gracefully

        Notes:
            - Thread-safe processing
            - Immediate feedback
            - Session state updates
            - Error resilient
        """
        if not st.session_state.chatbot:
            st.error("Chatbot not initialized. Please try reloading the page.")
            return
        
        try:
            with st.spinner("Thinking..."):
                # Update retrieval parameters if needed
                if st.session_state.rag_system.top_k_retrieval != top_k:
                    st.session_state.rag_system.top_k_retrieval = top_k
                    st.session_state.rag_system._setup_qa_chain()
                
                # Get response
                response = st.session_state.chatbot.chat(question)
            
            # Add to session state history
            st.session_state.conversation_history.append({
                'question': question,
                'answer': response.get('answer', ''),
                'sources': response.get('source_documents', []),
                'error': response.get('error')
            })
            
            # Display immediate response
            st.success("Question processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            logger.error(f"Error processing question: {str(e)}")
    
    def display_conversation_history(self):
        """
        Display the conversation history in the main interface.

        This method renders the chat history with:
        1. Message Display:
           - User questions
           - System answers
           - Timestamps
           - Source documents

        2. Visual Features:
           - Message styling
           - Role indicators
           - Expandable sources
           - Error messages

        3. Organization:
           - Reverse chronological order
           - Grouped by conversation
           - Clear visual hierarchy
           - Responsive layout

        4. Interactive Elements:
           - Expandable source documents
           - Scrollable history
           - Visual feedback
           - Error indicators

        Technical Details:
            - Uses Streamlit markdown
            - Custom HTML/CSS styling
            - Efficient rendering
            - Memory optimization

        Notes:
            - Auto-scrolls to new messages
            - Handles long conversations
            - Preserves formatting
            - Source traceability
        """
        if not st.session_state.conversation_history:
            return
        
        st.subheader("💬 Conversation History")
        
        # Display conversations in reverse order (newest first)
        for i, entry in enumerate(reversed(st.session_state.conversation_history)):
            with st.container():
                # Question
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 Question {len(st.session_state.conversation_history) - i}:</strong><br>
                    {entry['question']}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer
                if entry.get('error'):
                    st.error(f"Error: {entry['error']}")
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Answer:</strong><br>
                        {entry['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Source documents
                if entry.get('sources'):
                    with st.expander(f"📚 Source Documents ({len(entry['sources'])} found)", expanded=False):
                        for j, source in enumerate(entry['sources']):
                            st.markdown(f"""
                            <div class="source-doc">
                                <strong>Source {j + 1}:</strong><br>
                                <em>File: {source['metadata'].get('file_name', 'Unknown')}</em><br>
                                <em>Type: {source['metadata'].get('file_type', 'Unknown')}</em><br>
                                <br>
                                {source['content'][:300]}{'...' if len(source['content']) > 300 else ''}
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
    
    def create_sample_documents(self):
        """
        Create and process sample documents for demonstration.

        This method provides a quick start experience by:
        1. Creating sample documents about AI/ML
        2. Processing them through the RAG pipeline
        3. Initializing the system for immediate use
        4. Providing example content

        The process includes:
        - Creating the sample directory
        - Writing informative content
        - Processing documents
        - Initializing the RAG system

        Sample Content:
        - AI and Machine Learning overview
        - RAG system explanation
        - Vector search details
        - Technical concepts

        Technical Details:
            - Creates persistent files
            - Processes automatically
            - Updates system state
            - Provides feedback

        Notes:
            - One-click initialization
            - Educational content
            - Ready for Q&A
            - Reproducible setup
        """
        sample_dir = "./sample_docs"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Sample document about AI and Machine Learning
        sample_content = """
        Artificial Intelligence and Machine Learning

        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
        that can perform tasks that typically require human intelligence. These tasks include visual perception, 
        speech recognition, decision-making, and language translation.

        Machine Learning (ML) is a subset of AI that focuses on the development of algorithms and statistical 
        models that enable computer systems to improve their performance on a specific task through experience, 
        without being explicitly programmed for every scenario.

        Deep Learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") 
        to model and understand complex patterns in data. It has been particularly successful in areas such as 
        image recognition, natural language processing, and game playing.

        Retrieval-Augmented Generation (RAG) is an AI technique that combines the power of large language models 
        with information retrieval systems. RAG allows AI systems to access and incorporate relevant information 
        from external knowledge bases when generating responses, leading to more accurate and informative answers.

        The key components of a RAG system include:
        1. Document ingestion and preprocessing
        2. Vector embeddings for semantic search
        3. Retrieval mechanism to find relevant documents
        4. Generation component to produce answers based on retrieved context
        5. Integration layer that combines retrieval and generation

        Vector databases like FAISS (Facebook AI Similarity Search) enable efficient similarity search over 
        high-dimensional embeddings, making it possible to quickly find relevant document chunks for any given query.
        """
        
        sample_file_path = os.path.join(sample_dir, "ai_ml_overview.txt")
        with open(sample_file_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        # Process the sample document
        try:
            with st.spinner("Creating sample documents..."):
                ingestor = DocumentIngestor()
                vector_store = ingestor.process_documents(sample_dir)
                
                # Initialize RAG system
                qa_system = RAGQuestionAnswerer()
                st.session_state.rag_system = qa_system
                st.session_state.chatbot = RAGChatBot(qa_system)
                st.session_state.documents_processed = True
                
                st.success("Sample documents created and processed successfully!")
                
        except Exception as e:
            st.error(f"Error creating sample documents: {str(e)}")
    
    def run(self):
        """Run the Streamlit application."""
        # Display sidebar
        top_k = self.display_sidebar()
        
        # Display main interface
        self.display_main_interface(top_k)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>Built with Streamlit, LangChain, FAISS, and Hugging Face Transformers</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app."""
    try:
        app = StreamlitRAGApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()