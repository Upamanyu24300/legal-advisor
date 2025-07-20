import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Constants
CHROMA_DIR = "chroma_db"

def get_embeddings_model():
    """Initialize and return the HuggingFace embeddings model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return embeddings

def load_vector_store():
    """Load the existing vector store"""
    embeddings = get_embeddings_model()
    
    if not os.path.exists(CHROMA_DIR):
        raise ValueError(f"Vector store directory {CHROMA_DIR} does not exist. Please run ingest.py first.")
    
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    return vector_store

def create_rag_chain(retriever):
    """Create a RAG chain with the retriever and LLM"""
    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Create the prompt with conversation history context
    system_template = """You are an expert legal assistant specializing in Indian law. 
You MUST ONLY answer questions related to Indian law, legal matters, the Indian Constitution, legal rights, court cases, and legal procedures in India.
For any questions not related to Indian legal matters, politely inform the user that you can only assist with Indian legal topics.

Use the following pieces of context to answer the user's question about Indian legal matters.
Prioritize information from the provided context when available.
If the context doesn't contain the specific information needed, you can use your general knowledge about Indian law to provide a helpful response.
When using information from the context, cite the specific legal document, case, or section of the constitution you're referencing.
When using your general knowledge, clearly indicate this in your response.

Previous conversation:
{chat_history}

Context:
{context}

Question: {input}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
    ])
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the RAG chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain