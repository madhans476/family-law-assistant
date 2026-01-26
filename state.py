"""
Enhanced state.py with reasoning and explanation support.
"""

from typing import TypedDict, List, Dict, Optional, Literal, Tuple, Any
from langgraph.graph import MessagesState

class FamilyLawState(MessagesState):
    """Enhanced state for the family law assistant graph with explainability features."""
    
    # User query
    query: str
    
    # Analysis phase
    root_query: Optional[str] = None
    user_intent: Optional[str] = None
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
    current_question_target: Optional[str] = None
    
    # Re-validation support
    revalidation_mode: bool = False
    revalidation_count: int = 0
    
    # Update/correction handling
    is_update: bool = False
    update_type: Optional[Literal["correction", "addition", "clarification"]] = None
    previous_response_id: Optional[str] = None
    
    # Retrieval results
    retrieved_chunks: List[Dict] = []
    
    # Generated response
    response: str = ""
    
    # Explainability features
    reasoning_steps: List[Dict[str, Any]] = []  # Chain of reasoning
    precedent_explanations: List[Dict[str, Any]] = []  # Why each precedent matters
    include_reasoning: bool = True  # Whether to generate reasoning
    
    # Outcome prediction
    prediction: Optional[Dict] = None
    include_prediction: bool = True
    
    # Metadata
    sources: List[Dict] = []
    message_type: Optional[Literal["clarification", "information_gathering", "final_response", "update_response"]] = None
    
    # History management
    conversation_id: Optional[str] = None
    
    # Session tracking
    session_phase: Optional[Literal["initial", "gathering", "validating", "responding", "updating"]] = "initial"
    total_interactions: int = 0
    
    # Error handling
    last_error: Optional[str] = None
    retry_count: int = 0