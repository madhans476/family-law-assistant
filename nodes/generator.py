"""
Enhanced Generator Node - Generates comprehensive legal advice.

This version incorporates collected case information for more accurate responses.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict
from state import FamilyLawState
import os
import logging

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        task="conversational",
    )
)

SYSTEM_PROMPT = """You are a senior Indian family law attorney with 20+ years of experience. 
Your role is to provide clear, actionable legal advice based on case information and relevant legal precedents.

GUIDELINES:
1. Provide practical, step-by-step advice
2. Reference relevant Indian laws, sections, and precedents
3. Explain legal terms in simple English (Flesch Reading Ease score ≥ 55)
4. Be empathetic and also professional - family law matters are emotionally charged
5. Prioritize safety in domestic violence cases
6. Clarify this is informational advice, not a substitute for personalized legal representation
7. Structure your response with clear sections:
   - Immediate Actions
   - Legal Options Available
   - Relevant Laws & Precedents
   - Next Steps
8. Never make up information - only use provided context and case details

IMPORTANT: Format your response professionally with proper sections and bullet points where appropriate."""

def format_context(retrieved_chunks: list) -> str:
    """Format retrieved chunks into structured context."""
    if not retrieved_chunks:
        return "No relevant precedents found."
    
    context_parts = ["RELEVANT LEGAL PRECEDENTS AND CASES:\n"]
    
    for i, chunk in enumerate(retrieved_chunks[:5], 1):  # Limit to top 5
        context_parts.append(f"\n[Precedent {i}]")
        context_parts.append(f"Title: {chunk['metadata']['title']}")
        context_parts.append(f"Category: {chunk['metadata']['category']}")
        context_parts.append(f"Relevance: {chunk['score']:.2%}")
        context_parts.append(f"Content: {chunk['content'][:500]}...")  # Truncate long content
        context_parts.append("")
    
    return "\n".join(context_parts)

def format_case_info(info_collected: Dict, case_type: str) -> str:
    """Format collected case information."""
    if not info_collected:
        return "Limited case information available."
    
    case_summary = [f"CASE TYPE: {case_type.upper().replace('_', ' ')}\n"]
    case_summary.append("CASE DETAILS:")
    
    for key, value in info_collected.items():
        case_summary.append(f"- {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(case_summary)

def generate_response(state: FamilyLawState) -> Dict:
    """
    Generate comprehensive legal advice based on case information and retrieved context.
    """
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    messages = state.get("messages", [])
    info_collected = state.get("info_collected", {})
    case_type = state.get("case_type", "general")
    
    # Validate we have information
    if not retrieved_chunks:
        logger.warning("No retrieved chunks available for generation")
        return {
            "response": "I apologize, but I couldn't find sufficient relevant information in the legal database to provide comprehensive advice for your specific situation. Please consider consulting with a family law attorney directly.",
            "messages": messages
        }
    
    # Format context and case information
    legal_context = format_context(retrieved_chunks)
    case_information = format_case_info(info_collected, case_type)
    
    # Build conversation with history
    conversation = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Add relevant previous context (last 4 messages)
    if messages:
        conversation.extend(messages[-4:])
    
    # Construct the prompt
    prompt = f"""Based on the collected case information and relevant legal precedents, provide comprehensive legal advice.

{case_information}

{legal_context}

CURRENT QUERY: {query}

Provide a detailed, well-structured response that:
1. Addresses the immediate concerns
2. Outlines available legal options
3. References specific Indian laws and sections
4. Provides clear next steps
5. Uses simple, empathetic language

YOUR RESPONSE:"""
    
    conversation.append(HumanMessage(content=prompt))
    
    try:
        # Generate response
        response = llm.invoke(conversation)
        
        # Add disclaimer if not already present
        response_content = response.content
        if "not a substitute for legal advice" not in response_content.lower():
            response_content += "\n\n---\n*Disclaimer: This information is for educational purposes only and does not constitute legal advice. Please consult with a qualified family law attorney for personalized legal guidance.*"
        
        logger.info(f"Successfully generated response for case type: {case_type}")
        
        return {
            "response": response_content,
            "messages": conversation + [response]
        }
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {
            "response": f"I apologize, but I encountered an error while generating advice. Please try rephrasing your question or contact support. Error: {str(e)}",
            "messages": messages
        }