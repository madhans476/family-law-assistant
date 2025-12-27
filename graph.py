"""
LangGraph workflow with integrated debug logging.

The logging functions are called INSIDE the node functions, not as separate nodes.
"""

from langgraph.graph import StateGraph, START, END
from state import FamilyLawState
from nodes.query_analyzer import QueryAnalyzer
from nodes.information_gatherer import InformationGatherer
from nodes.retriever import retrieve_documents
from nodes.generator import generate_response
import logging

# Import debug logging utilities
from nodes.logs import log_state_transition, log_gathering_iteration

logger = logging.getLogger(__name__)


def analyze_query_node(state: FamilyLawState) -> FamilyLawState:
    """
    Analyze ONLY the initial query. Skip if in gathering phase.
    """
    # Save state before for logging
    state_before = dict(state)
    
    # CRITICAL: Don't analyze follow-up responses
    if state.get("in_gathering_phase", False):
        logger.info("Skipping analysis - already in gathering phase")
        return state
    
    if state.get("analysis_complete", False):
        logger.info("Skipping analysis - already complete")
        return state
    
    try:
        logger.info("=== ANALYZING INITIAL QUERY ===")
        agent = QueryAnalyzer()
        response = agent.analyze_query(state)
        
        # Update state
        state["user_intent"] = response.get("user_intent")
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["has_sufficient_info"] = response.get("has_sufficient_info", False)
        state["info_collected"] = response.get("info_collected", {})
        state["analysis_complete"] = True
        
        # Check intent confidence
        intent_confidence = response.get("intent_confidence", "high")
        if intent_confidence == "low" or not response.get("user_intent"):
            logger.info("Low confidence - requesting clarification")
            state["needs_clarification"] = True
            state["clarification_question"] = "Could you please provide more details about your legal situation? For example, is this related to divorce, domestic violence, child custody, or another family law matter?"
        else:
            state["needs_clarification"] = False
            
            if not state["info_needed_list"]:
                logger.info("No info needed - ready for retrieval")
                state["has_sufficient_info"] = True
                state["in_gathering_phase"] = False
            else:
                logger.info(f"Need to gather {len(state['info_needed_list'])} items")
                state["in_gathering_phase"] = True
                state["gathering_step"] = 0
        
        # Log state transition
        log_state_transition("analyze_query", state_before, state)
        
        return state
        
    except Exception as e:
        logger.error(f"Query Analyzer failed: {e}")
        state["has_sufficient_info"] = True
        state["analysis_complete"] = True
        state["in_gathering_phase"] = False
        return state


def gather_information_node(state: FamilyLawState) -> FamilyLawState:
    """
    Gather information iteratively with debug logging.
    """
    # Save state before for logging
    state_before = dict(state)
    
    try:
        # Log start of iteration
        log_gathering_iteration(
            state.get("gathering_step", 0),
            state,
            "START"
        )
        
        logger.info("=== GATHERING INFORMATION ===")
        gatherer = InformationGatherer()
        response = gatherer.gather_next_information(state)
        
        # CRITICAL: Update state with new values
        state["info_collected"] = response.get("info_collected", {})
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["follow_up_question"] = response.get("follow_up_question")
        state["needs_more_info"] = response.get("needs_more_info", False)
        state["gathering_step"] = response.get("gathering_step", 0)
        state["current_question_target"] = response.get("current_question_target")
        
        # Check completion
        if not state["needs_more_info"]:
            logger.info("✓ Gathering complete!")
            state["has_sufficient_info"] = True
            state["in_gathering_phase"] = False
        
        # Log end of iteration
        log_gathering_iteration(
            state.get("gathering_step", 0),
            state,
            "END"
        )
        
        # Log state transition
        log_state_transition("gather_information", state_before, state)
        
        return state
        
    except Exception as e:
        logger.error(f"Information Gatherer failed: {e}")
        state["has_sufficient_info"] = True
        state["in_gathering_phase"] = False
        state["needs_more_info"] = False
        return state


def route_after_analysis(state: FamilyLawState) -> str:
    """Route after initial query analysis."""
    if state.get("needs_clarification", False):
        logger.info("→ Routing to clarification")
        return "clarify"
    
    has_sufficient_info = state.get("has_sufficient_info", False)
    info_needed = state.get("info_needed_list", [])
    
    if has_sufficient_info or not info_needed:
        logger.info("→ Routing to retrieval (sufficient info)")
        return "retrieve"
    else:
        logger.info(f"→ Routing to gather_info (need {len(info_needed)} items)")
        return "gather_info"


def route_after_gathering(state: FamilyLawState) -> str:
    """Route after information gathering attempt."""
    needs_more_info = state.get("needs_more_info", False)
    
    if needs_more_info:
        logger.info("→ Routing to ask_question (more info needed)")
        return "ask_question"
    else:
        logger.info("→ Routing to retrieval (gathering complete)")
        return "retrieve"


def format_clarification_response(state: FamilyLawState) -> dict:
    """Format clarification request."""
    clarification = state.get(
        "clarification_question",
        "Could you please clarify your legal situation?"
    )
    
    return {
        "response": clarification,
        "sources": [],
        "message_type": "clarification"
    }


def format_follow_up_response(state: FamilyLawState) -> dict:
    """Format follow-up question with progress."""
    follow_up = state.get(
        "follow_up_question",
        "Could you provide more details?"
    )
    
    info_collected = state.get("info_collected", {})
    info_needed = state.get("info_needed_list", [])
    
    return {
        "response": follow_up,
        "sources": [],
        "message_type": "information_gathering",
        "info_collected": info_collected,
        "info_needed": info_needed
    }


def create_graph():
    """
    Create the family law assistant graph.
    
    The debug logging is integrated INSIDE the node functions,
    not added as separate workflow nodes.
    """
    
    workflow = StateGraph(FamilyLawState)
    
    # Add nodes (logging is INSIDE these functions)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("clarify", format_clarification_response)
    workflow.add_node("gather_info", gather_information_node)  # Has logging inside
    workflow.add_node("ask_question", format_follow_up_response)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_response)
    
    # Edges (same as before)
    workflow.add_edge(START, "analyze_query")
    
    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "clarify": "clarify",
            "gather_info": "gather_info",
            "retrieve": "retrieve"
        }
    )
    
    workflow.add_edge("clarify", END)
    
    workflow.add_conditional_edges(
        "gather_info",
        route_after_gathering,
        {
            "ask_question": "ask_question",
            "retrieve": "retrieve"
        }
    )
    
    workflow.add_edge("ask_question", END)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()
    logger.info("✓ Graph compiled successfully with debug logging")
    
    return app


family_law_app = create_graph()