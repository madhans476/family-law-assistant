from typing import TypedDict, List, Dict, Optional
from langgraph.graph import MessagesState

class FamilyLawState(MessagesState):
    """State for the family law assistant graph."""
    
    # User query
    query: str
    
    # Retrieval results
    retrieved_chunks: List[Dict]
    
    # Generated response
    response: str
    
    # Metadata
    sources: List[Dict]
    
    # History management
    conversation_id: Optional[str]