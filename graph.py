"""
Enhanced LangGraph workflow with iterative information gathering.

Flow:
1. analyze_query: Understand user intent (with clarification loop if needed)
2. gather_info: Ask questions iteratively until all info is collected
3. retrieve: Get relevant legal documents
4. generate: Provide comprehensive legal advice

All intermediate steps are visible in the frontend.
"""

from langgraph.graph import StateGraph, START, END
from state import FamilyLawState
from nodes.query_analyzer import QueryAnalyzer
from nodes.information_gatherer import InformationGatherer
from nodes.retriever import retrieve_documents
from nodes.generator import generate_response
import logging

logger = logging.getLogger(__name__)


def analyze_query_node(state: FamilyLawState) -> FamilyLawState:
    """
    Analyze the user query to determine intent and information needs.
    Only runs on the initial query, not on follow-up responses.
    """
    try:
        # Check if this is a follow-up response during info gathering
        if state.get("in_gathering_phase", False):
            logger.info("Skipping analysis - in gathering phase")
            return state
        
        agent = QueryAnalyzer()
        response = agent.analyze_query(state)
        
        # Update state with analysis results
        state["user_intent"] = response.get("user_intent")
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["has_sufficient_info"] = response.get("has_sufficient_info", False)
        state["info_collected"] = response.get("info_collected", {})
        state["analysis_complete"] = True
        
        # Check if intent is unclear
        intent_confidence = response.get("intent_confidence", "high")
        if intent_confidence == "low" or not response.get("user_intent"):
            logger.info("User intent unclear, requesting clarification")
            state["needs_clarification"] = True
            state["clarification_question"] = "I'd like to help you better. Could you please provide more details about your legal situation? For example, is this related to divorce, domestic violence, child custody, or another family law matter?"
        else:
            state["needs_clarification"] = False
            
            # If no info needed, mark as ready to retrieve
            if not state["info_needed_list"]:
                logger.info("No information needed, proceeding to retrieval")
                state["has_sufficient_info"] = True
                state["in_gathering_phase"] = False
            else:
                # Enter gathering phase
                state["in_gathering_phase"] = True
        
        return state
        
    except Exception as e:
        logger.error(f"Query Analyzer failed: {e}")
        state["has_sufficient_info"] = True
        state["analysis_complete"] = True
        state["in_gathering_phase"] = False
        return state


def gather_information_node(state: FamilyLawState) -> FamilyLawState:
    """
    Gather information iteratively by asking one question at a time.
    """
    try:
        gatherer = InformationGatherer()
        response = gatherer.gather_next_information(state)
        
        # Update state with gathering results
        state["info_collected"] = response.get("info_collected", state.get("info_collected", {}))
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["follow_up_question"] = response.get("follow_up_question")
        state["needs_more_info"] = response.get("needs_more_info", False)
        
        # Check if gathering is complete
        if not state["needs_more_info"]:
            logger.info("Information gathering complete")
            state["has_sufficient_info"] = True
            state["in_gathering_phase"] = False
        
        return state
        
    except Exception as e:
        logger.error(f"Information Gatherer failed: {e}")
        # On error, proceed to retrieval with what we have
        state["has_sufficient_info"] = True
        state["in_gathering_phase"] = False
        state["needs_more_info"] = False
        return state


def route_after_analysis(state: FamilyLawState) -> str:
    """
    Route after query analysis.
    
    Returns:
        - "clarify" if intent is unclear
        - "gather_info" if more information is needed
        - "retrieve" if sufficient information is available
    """
    if state.get("needs_clarification", False):
        logger.info("Routing to clarification")
        return "clarify"
    
    has_sufficient_info = state.get("has_sufficient_info", False)
    info_needed = state.get("info_needed_list", [])
    
    if has_sufficient_info or not info_needed:
        logger.info("Sufficient information, proceeding to retrieval")
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


def format_clarification_response(state: FamilyLawState) -> dict:
    """
    Format the clarification request as a response.
    """
    clarification = state.get(
        "clarification_question",
        "I'd like to help you better. Could you please clarify what specific legal issue you're facing?"
    )
    
    return {
        "response": clarification,
        "sources": [],
        "message_type": "clarification"
    }


def format_follow_up_response(state: FamilyLawState) -> dict:
    """
    Format the follow-up question as a response.
    """
    follow_up = state.get(
        "follow_up_question",
        "Could you provide more details about your situation?"
    )
    
    # Add context about what information we're collecting
    info_collected = state.get("info_collected", {})
    info_needed = state.get("info_needed_list", [])
    
    progress_info = ""
    if info_collected:
        progress_info = f"\n\n📋 Information collected: {len(info_collected)} items"
    if info_needed:
        progress_info += f"\n📝 Still needed: {len(info_needed)} items"
    
    return {
        "response": follow_up + progress_info,
        "sources": [],
        "message_type": "information_gathering",
        "info_collected": info_collected,
        "info_needed": info_needed
    }


def create_graph():
    """Create the enhanced family law assistant graph with iterative flow."""
    
    # Initialize graph
    workflow = StateGraph(FamilyLawState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("clarify", format_clarification_response)
    workflow.add_node("gather_info", gather_information_node)
    workflow.add_node("ask_question", format_follow_up_response)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_response)
    
    # Add edges with conditional routing
    # Start -> Analyze Query
    workflow.add_edge(START, "analyze_query")
    
    # After analysis, route to clarify/gather/retrieve
    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "clarify": "clarify",
            "gather_info": "gather_info",
            "retrieve": "retrieve"
        }
    )
    
    # After clarification, end (wait for user response)
    workflow.add_edge("clarify", END)
    
    # After gathering info, either ask question or retrieve
    workflow.add_conditional_edges(
        "gather_info",
        route_after_gathering,
        {
            "ask_question": "ask_question",
            "retrieve": "retrieve"
        }
    )
    
    # After asking question, end (wait for user response)
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