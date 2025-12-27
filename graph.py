"""
Enhanced LangGraph workflow with intelligent query analysis and information gathering.

Flow:
1. analyze_query: Understand user intent and information needs
2. gather_info: Ask follow-up questions if needed
3. retrieve: Get relevant legal documents
4. generate: Provide comprehensive legal advice
"""

from langgraph.graph import StateGraph, START, END
from state import FamilyLawState
from nodes.query_analyzer import QueryAnalyzer
from nodes.information_gatherer import gather_information
from nodes.retriever import retrieve_documents
from nodes.generator import generate_response
import logging

logger = logging.getLogger(__name__)


def analyze_query_node(state: FamilyLawState) -> FamilyLawState:
    """
    Analyze the user query to determine intent and information needs.
    """

    try:
        agent = QueryAnalyzer()
        response = agent.analyze_query(state)
        state["user_intent"] = response.get("user_intent")
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["has_sufficient_info"] = response.get("has_sufficient_info", False)
        state["info_collected"] = response.get("info_collected", {})
        if not response.get("info_needed_list", []):
            logger.info("No information needed, proceeding to retrieval")
            state["has_sufficient_info"] = True
            state["needs_more_info"] = False
        return state
    except Exception as e:
        print(f"[ERROR] Query Analyzer failed: {e}")
        state["has_sufficient_info"] = True
        return state
    

def route_after_analysis(state: FamilyLawState) -> str:
    """
    Route after query analysis based on information sufficiency.
    
    Returns:
        - "gather_info" if more information is needed
        - "retrieve" if sufficient information is available
    """
    has_sufficient_info = state.get("has_sufficient_info", False)
    info_needed = state.get("info_needed_list", [])
    
    if has_sufficient_info or not info_needed:
        logger.info("Sufficient information available, proceeding to retrieval")
        return "retrieve"
    else:
        logger.info(f"Need to gather {len(info_needed)} pieces of information")
        return "gather_info"

def route_after_gathering(state: FamilyLawState) -> str:
    """
    Route after information gathering.
    
    Returns:
        - "ask_question" if more information is needed
        - "retrieve" if sufficient information is collected
    """
    needs_more_info = state.get("needs_more_info", False)
    
    if needs_more_info:
        logger.info("Asking follow-up question")
        return "ask_question"
    else:
        logger.info("Information gathering complete, proceeding to retrieval")
        return "retrieve"

def format_follow_up_response(state: FamilyLawState) -> dict:
    """
    Format the follow-up question as a response.
    """
    follow_up = state.get("follow_up_question", "Could you provide more details about your situation?")
    
    return {
        "response": follow_up,
        "sources": []
    }

def create_graph():
    """Create the enhanced family law assistant graph."""
    
    # Initialize graph
    workflow = StateGraph(FamilyLawState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("gather_info", gather_information)
    workflow.add_node("ask_question", format_follow_up_response)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_response)
    
    # Add edges with conditional routing
    # Start -> Analyze Query
    workflow.add_edge(START, "analyze_query")
    
    # After analysis, either gather info or retrieve
    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "gather_info": "gather_info",
            "retrieve": "retrieve"
        }
    )
    
    # After gathering info, either ask question or retrieve
    workflow.add_conditional_edges(
        "gather_info",
        route_after_gathering,
        {
            "ask_question": "ask_question",
            "retrieve": "retrieve"
        }
    )
    
    # After asking question, end (user needs to respond)
    workflow.add_edge("ask_question", END)
    
    # Normal flow: retrieve -> generate -> end
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile graph
    app = workflow.compile()
    
    logger.info("Family law assistant graph compiled successfully")
    
    return app

# Create the compiled graph
family_law_app = create_graph()