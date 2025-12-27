"""
Enhanced State Management for Family Law Assistant.

This state supports multi-turn conversations with structured information collection
and visibility of all intermediate steps.
"""

from typing import TypedDict, List, Dict, Optional, Literal
from langgraph.graph import MessagesState

class FamilyLawState(MessagesState):
    """Enhanced state for the family law assistant graph."""
    
    # User query
    query: str
    
    # Analysis phase
    user_intent: Optional[str]  # What user wants to achieve
    analysis_complete: bool  # Whether initial analysis is done
    needs_clarification: bool  # Whether user intent needs clarification
    clarification_question: Optional[str]  # Question to clarify intent
    
    # Information collection phase
    in_gathering_phase: bool = False  # Whether we're currently gathering info
    has_sufficient_info: bool = False  # Whether we have enough info to answer
    info_collected: Dict[str, str] = {}  # Information already gathered {key: value}
    info_needed_list: List[str] = []  # List of specific information still needed
    needs_more_info: bool = True  # Whether to ask follow-up questions
    follow_up_question: Optional[str] = None  # The question to ask
    gathering_step: int = 0  # Current step in gathering process
    
    # Retrieval results
    retrieved_chunks: List[Dict]
    
    # Generated response
    response: str
    
    # Metadata
    sources: List[Dict]
    message_type: Optional[Literal["clarification", "information_gathering", "final_response"]]
    
    # History management
    conversation_id: Optional[str]