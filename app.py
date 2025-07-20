import streamlit as st
from utils import load_vector_store, create_rag_chain
import time

# Set page configuration
st.set_page_config(
    page_title="Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Force dark mode for better visibility
st.markdown("""
<script>
    var body = window.parent.document.querySelector('body');
    body.classList.add('dark');
</script>
""", unsafe_allow_html=True)

# Custom CSS for better styling with dark mode support
st.markdown("""
<style>
.main-header {
    font-family: 'Helvetica', sans-serif;
    text-align: center;
    color: #1E3A8A;
    padding: 1.5rem 0;
    border-bottom: 2px solid #E5E7EB;
    margin-bottom: 2rem;
    background-color: #F3F4F6;
    border-radius: 10px;
}
.description {
    text-align: center;
    margin-bottom: 2rem;
    font-size: 1.1rem;
    color: #4B5563;
}
.stChatMessage {
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stChatMessageContent {
    border-radius: 12px;
}
.user-message {
    background-color: #E9F2FF;
    color: #1F2937 !important; /* Dark text for light background */
}
.assistant-message {
    background-color: #F0FFF4;
    color: #1F2937 !important; /* Dark text for light background */
}
/* Dark mode specific styles */
@media (prefers-color-scheme: dark) {
    .main-header {
        color: #93C5FD;
        background-color: #1F2937;
        border-bottom: 2px solid #374151;
    }
    .description {
        color: #D1D5DB;
    }
    .user-message {
        background-color: #1E40AF;
        color: #F3F4F6 !important;
    }
    .assistant-message {
        background-color: #065F46;
        color: #F3F4F6 !important;
    }
    .sidebar-header {
        color: #93C5FD;
        border-bottom: 1px solid #374151;
    }
    .sidebar-content {
        color: #D1D5DB;
    }
    .disclaimer {
        background-color: #78350F;
        color: #FBBF24;
    }
}
.sidebar-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1E3A8A;
    margin-bottom: 1rem;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 0.5rem;
}
.sidebar-content {
    color: #4B5563;
    font-size: 0.95rem;
}
.disclaimer {
    background-color: #FEF3C7;
    padding: 1rem;
    border-radius: 8px;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #92400E;
}
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header"><h1>🇮🇳 Indian Legal Assistant</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="description">This chatbot provides guidance on Indian legal matters based on the Constitution of India, landmark Supreme Court cases, and other legal documents.</div>', unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vector store and create retriever
@st.cache_resource
def load_retriever():
    try:
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

retriever = load_retriever()

# Function to format chat history for the RAG chain
def format_chat_history(messages):
    chat_history = ""
    for msg in messages:
        role = "Human" if msg["role"] == "user" else "Assistant"
        chat_history += f"{role}: {msg['content']}\n\n"
    return chat_history

# Display chat messages with enhanced styling
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='{role_class}' style='color: inherit;'>{message['content']}</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about Indian legal matters..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if retriever is None:
            st.markdown("Vector store not found. Please run the ingest.py script first.")
        else:
            with st.spinner("Thinking..."):
                # Create RAG chain with chat history context
                rag_chain = create_rag_chain(retriever)
                
                # Format chat history for context
                chat_history = format_chat_history(st.session_state.messages[:-1])  # Exclude current message
                
                try:
                    # Invoke the chain with the prompt and chat history
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                    answer = response["answer"]
                    
                    # Create a placeholder for the typing animation
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Simulate typing with a simple animation
                    for chunk in answer.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(f"<div class='assistant-message' style='color: inherit;'>{full_response}▌</div>", unsafe_allow_html=True)
                        time.sleep(0.05)  # Adjust speed as needed
                    
                    # Display the final answer
                    message_placeholder.markdown(f"<div class='assistant-message' style='color: inherit;'>{answer}</div>", unsafe_allow_html=True)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {e}")

# Sidebar with information and enhanced styling
with st.sidebar:
    st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">This legal assistant uses Retrieval-Augmented Generation (RAG) to provide accurate information about Indian law, drawing from:<br><br>• <b>The Constitution of India</b><br>• <b>Supreme Court landmark judgments</b><br>• <b>High Court cases</b><br>• <b>Legal precedents and statutes</b></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">• Answers questions about Indian law and legal rights<br>• Provides information on legal procedures<br>• Explains constitutional provisions<br>• References relevant court cases<br>• Maintains conversation context</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> This chatbot provides general legal information and not legal advice. Always consult with a qualified legal professional for specific legal matters.</div>', unsafe_allow_html=True)
    
    # Clear chat button with better styling
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Clear Chat", type="primary"):
            st.session_state.messages = []
            st.rerun()