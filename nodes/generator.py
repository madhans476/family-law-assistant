"""
Fixed Generator Node - Generates complete legal advice without truncation.

Key fixes:
1. Increased max_new_tokens to allow longer responses
2. Better context formatting to reduce input token usage
3. Streamlined prompt to focus on essential information
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict
from state import FamilyLawState
import os
import logging

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM with higher token limit
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        task="conversational",
        max_new_tokens=2048,  # INCREASED from default (~512) to 2048
        temperature=0.7,
    )
)

SYSTEM_PROMPT = """You are a senior Indian family law attorney with 20+ years of experience. 
Provide clear, actionable legal advice based on case information and relevant legal precedents.

GUIDELINES:
1. Provide practical, step-by-step advice
2. Reference relevant Indian laws, sections, and precedents
3. Explain legal terms simply (Flesch Reading Ease ≥ 55)
4. Be empathetic and professional
5. Prioritize safety in domestic violence cases
6. Structure response with clear sections:
   - Immediate Actions
   - Legal Options Available
   - Relevant Laws & Precedents
   - Next Steps
7. Only use provided context - never make up information

IMPORTANT: Keep your response complete and comprehensive. Do not truncate."""

def format_context(retrieved_chunks: list) -> str:
    """Format retrieved chunks efficiently to save tokens."""
    if not retrieved_chunks:
        return "No relevant precedents found."
    
    context_parts = ["RELEVANT LEGAL PRECEDENTS:\n"]
    
    # Limit to top 3 most relevant to save tokens
    for i, chunk in enumerate(retrieved_chunks[:3], 1):
        context_parts.append(f"\n[Precedent {i}] ({chunk['score']:.0%} relevance)")
        context_parts.append(f"Title: {chunk['metadata']['title']}")
        context_parts.append(f"Category: {chunk['metadata']['category']}")
        # Use first 400 chars instead of 500 to save tokens
        content = chunk['content'][:400]
        if len(chunk['content']) > 400:
            content += "..."
        context_parts.append(f"Content: {content}")
        context_parts.append("")
    
    return "\n".join(context_parts)

def format_case_info(info_collected: Dict, user_intent: str) -> str:
    """Format collected case information concisely."""
    if not info_collected:
        return "Limited case information available."
    
    case_summary = [f"CASE: {user_intent.upper()}\n"]
    case_summary.append("CLIENT INFORMATION:")
    
    for key, value in info_collected.items():
        case_summary.append(f"• {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(case_summary)

def generate_response(state: FamilyLawState) -> Dict:
    """
    Generate comprehensive legal advice.
    """
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    messages = state.get("messages", [])
    info_collected = state.get("info_collected", {})
    user_intent = state.get("user_intent", "legal advice")
    
    # Validate we have information
    if not retrieved_chunks:
        logger.warning("No retrieved chunks available for generation")
        return {
            "response": "I apologize, but I couldn't find sufficient relevant information in the legal database to provide comprehensive advice for your specific situation. Please consider consulting with a family law attorney directly for personalized guidance.",
            "messages": messages
        }
    
    # Format context and case information
    legal_context = format_context(retrieved_chunks)
    case_information = format_case_info(info_collected, user_intent)
    
    # Build conversation - only include recent history to save tokens
    conversation = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Only include last 2 exchanges to keep context manageable
    if messages:
        conversation.extend(messages[-4:])
    
    # Construct focused prompt
    prompt = f"""Provide comprehensive legal advice based on the case information and precedents below.

{case_information}

{legal_context}

CLIENT QUERY: {query}

Provide a COMPLETE, well-structured response with:
1. Immediate Actions (practical steps to take now)
2. Legal Options Available (specific remedies under Indian law)
3. Relevant Laws & Precedents (cite sections and acts)
4. Next Steps (clear action plan)

Use empathetic, simple language. Be thorough - this is important for the client's case.

YOUR COMPLETE RESPONSE:"""
    
    conversation.append(HumanMessage(content=prompt))
    
    try:
        logger.info("Generating response with max_tokens=2048")
        
        # Generate response
        response = llm.invoke(conversation)
        response_content = response.content
        
        # Check if response seems truncated
        if len(response_content) < 500:
            logger.warning(f"Response seems short: {len(response_content)} chars")
        
        # Add disclaimer if not present
        if "not a substitute for legal advice" not in response_content.lower():
            response_content += "\n\n---\n**Disclaimer**: This information is for educational purposes only and does not constitute legal advice. Please consult with a qualified family law attorney for personalized legal guidance."
        
        logger.info(f"Generated response: {len(response_content)} characters")
        
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