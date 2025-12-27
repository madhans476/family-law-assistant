"""
Enhanced State with tracking for current question target.
"""

from typing import TypedDict, List, Dict, Optional, Literal
from langgraph.graph import MessagesState

class FamilyLawState(MessagesState):
    """Enhanced state for the family law assistant graph."""
    
    # User query
    query: str
    
    # Analysis phase
    user_intent: Optional[str]
    analysis_complete: bool = False
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    
    # Information collection phase
    in_gathering_phase: bool = False
    has_sufficient_info: bool = False
    info_collected: Dict[str, str] = {}
    info_needed_list: List[str] = []
    needs_more_info: bool = True
    follow_up_question: Optional[str] = None
    gathering_step: int = 0
    current_question_target: Optional[str] = None  # Track what we're currently asking about
    
    # Retrieval results
    retrieved_chunks: List[Dict] = []
    
    # Generated response
    response: str = ""
    
    # Metadata
    sources: List[Dict] = []
    message_type: Optional[Literal["clarification", "information_gathering", "final_response"]] = None
    
    # History management
    conversation_id: Optional[str] = None