"""
Information Gatherer Node - Collects specific information based on query analysis.

This node receives the list of needed information from query_analyzer
and asks targeted questions to gather that information.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List
from state import FamilyLawState
import os
import logging
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        task="conversational",
    )
)

INFORMATION_GATHERER_PROMPT = """You are a compassionate Indian Family Law attorney conducting a client consultation.

SITUATION:
- User Intent: {user_intent}
- Information already collected: {info_collected}
- Information still needed: {info_needed}

YOUR TASK:
Ask ONE clear, empathetic, professional question to gather the MOST IMPORTANT missing information from the list(Information still needed section) above.

GUIDELINES:
1. Ask only ONE or TWO question at a time
2. Be empathetic and professional
3. Use simple, clear language
4. For domestic violence: prioritize safety questions
5. Make the question specific and easy to answer

CONVERSATION HISTORY:
{conversation_history}

USER'S LATEST MESSAGE:
{user_message}

YOUR QUESTION (one clear question only):"""

def format_info_needed(info_needed_list: List[str]) -> str:
    """Format the list of needed information."""
    
    formatted = []
    for item in info_needed_list:
        formatted.append(f"- {item.replace('_', ' ').title()}")
    return "\n".join(formatted)

def format_info_collected(info_collected: Dict) -> str:
    """Format collected information."""
    if not info_collected:
        return "No information collected yet."
    
    formatted = []
    for key, value in info_collected.items():
        formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
    return "\n".join(formatted)

def format_conversation_history(messages: List) -> str:
    """Format recent conversation history."""
    if not messages:
        return "No previous conversation."
    
    formatted = []
    for msg in messages[-4:]:  # Last 2 exchanges
        role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
        formatted.append(f"{role}: {msg.content[:600]}")
    return "\n".join(formatted)

def gather_information(state: FamilyLawState) -> Dict:
    """
    Generate a follow-up question based on needed information.
    
    Returns:
        Dict with:
        - needs_more_info: bool
        - follow_up_question: str (if needs_more_info is True)
    """
    query = state["query"]
    messages = state.get("messages", [])
    info_collected = state.get("info_collected", {})
    info_needed_list = state.get("info_needed_list", [])
    user_intent = state.get("user_intent", "legal advice")
    
    # If no information is needed, skip gathering
    if not info_needed_list:
        logger.info("No information needed, proceeding to retrieval")
        return {
            "needs_more_info": False
        }
    
    # # Safety check for domestic violence
    # if case_type == "domestic_violence" and info_needed_list:
    #     if any("safety" in item.lower() for item in info_needed_list):
    #         return {
    #             "needs_more_info": True,
    #             "follow_up_question": "Before we proceed, I need to ensure your safety. Are you currently in a safe location away from the person who harmed you?"
    #         }
    
    # Prepare prompt for LLM
    prompt_content = INFORMATION_GATHERER_PROMPT.format(
        user_intent=user_intent.replace("_", " ").title(),
        info_collected=format_info_collected(info_collected),
        info_needed=format_info_needed(info_needed_list),
        conversation_history=format_conversation_history(messages),
        user_message=query
    )
    
    try:
        # Get LLM response
        conversation = [
            SystemMessage(content="You are an empathetic family law attorney gathering case information."),
            HumanMessage(content=prompt_content)
        ]
        
        response = llm.invoke(conversation)
        follow_up_question = response.content.strip()
        
        # Remove any prefixes like "Question:" or "My question:"
        if ":" in follow_up_question:
            parts = follow_up_question.split(":", 1)
            if len(parts[0].split()) <= 3:  # Short prefix
                follow_up_question = parts[1].strip()
        
        logger.info(f"Generated follow-up question for {user_intent}")
        
        # Extract information from user's response if this is a follow-up
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message.__class__.__name__ == "HumanMessage":
                # Store the response
                if info_needed_list:
                    # Map the first needed info to the user's response
                    first_needed = info_needed_list[0]
                    info_collected[first_needed] = last_message.content
                    # Remove from needed list
                    info_needed_list = info_needed_list[1:]
        
        # Check if we still need more information
        needs_more = len(info_needed_list) > 0
        
        return {
            "needs_more_info": needs_more,
            "follow_up_question": follow_up_question if needs_more else None,
            "info_collected": info_collected,
            "info_needed_list": info_needed_list
        }
    
    except Exception as e:
        logger.error(f"Error generating follow-up question: {str(e)}")
        # Fallback question
        if info_needed_list:
            first_needed = info_needed_list[0].replace("_", " ").title()
            return {
                "needs_more_info": True,
                "follow_up_question": f"Could you please provide details about: {first_needed}?"
            }
        
        return {
            "needs_more_info": False
        }