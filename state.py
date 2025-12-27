"""
Enhanced State Management for Family Law Assistant.

This state supports multi-turn conversations with structured information collection.
"""

from typing import TypedDict, List, Dict, Optional, Literal
from langgraph.graph import MessagesState

class FamilyLawState(MessagesState):
    """Enhanced state for the family law assistant graph."""
    
    # User query
    query: str
    
    # Case identification (from query_analyzer)
    # case_type: Optional[str]  # divorce, domestic_violence, dowry, custody, maintenance, general
    user_intent: Optional[str]  # What user wants to achieve
    
    # Information collection (from query_analyzer and information_gatherer)
    has_sufficient_info: bool  # Whether we have enough info to answer
    info_collected: Dict[str, str]  # Information already gathered
    info_needed_list: List[str]  # List of specific information still needed
    needs_more_info: bool  # Whether to ask follow-up questions
    follow_up_question: Optional[str]  # The question to ask
    
    # Retrieval results
    retrieved_chunks: List[Dict]
    
    # Generated response
    response: str
    
    # Metadata
    sources: List[Dict]
    
    # History management
    conversation_id: Optional[str]
    
    # Routing decision
    next_node: Optional[Literal["analyze", "gather_info", "retrieve", "generate"]]