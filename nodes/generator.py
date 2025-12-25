from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict
from state import FamilyLawState
import os

# Initialize LLM - Llama 3.1 8B Instruct from Hugging Face
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        task="conversational",
    )
)


SYSTEM_PROMPT = """You are a knowledgeable family law legal assistant. Your role is to provide accurate, helpful information about family law matters based on the retrieved case information and legal expertise.

Guidelines:
- Provide clear, practical advice based on the context provided
- Reference relevant cases or legal principles when applicable
- Be empathetic and professional, as family law matters are often sensitive
- If the retrieved information doesn't fully address the query, acknowledge this and provide general guidance
- Always clarify that this is informational and not a substitute for personalized legal advice
- Format your response clearly with proper structure

Never make up information. Only use the provided context."""

def format_context(retrieved_chunks):
    """Format retrieved chunks into context string."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Source {i}] (Category: {chunk['metadata']['category']})")
        context_parts.append(f"Title: {chunk['metadata']['title']}")
        context_parts.append(f"Content: {chunk['content']}")
        context_parts.append(f"Relevance Score: {chunk['score']:.3f}\n")
    
    return "\n".join(context_parts)

def generate_response(state: FamilyLawState) -> Dict:
    """
    Generate a response using Claude based on retrieved context.
    """
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    messages = state.get("messages", [])
    
    if not retrieved_chunks:
        return {
            "response": "I apologize, but I couldn't find relevant information in the family law database to answer your question. Please try rephrasing your question or ask about topics like divorce, custody, domestic violence, dowry, or marital cruelty."
        }
    
    # Format context
    context = format_context(retrieved_chunks)
    
    # Build conversation with history
    conversation = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Add previous messages (history)
    if messages:
        conversation.extend(messages)
    
    # Add current query with context
    user_message = f"""Based on the following retrieved information from family law cases, please answer the user's question:

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

Provide a comprehensive, helpful response based on the above context."""
    
    conversation.append(HumanMessage(content=user_message))
    
    # Generate response
    response = llm.invoke(conversation)
    
    return {
        "response": response.content,
        "messages": conversation + [response]
    }