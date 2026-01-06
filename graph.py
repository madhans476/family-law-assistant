"""
Enhanced graph.py with support for:
1. Post-gathering re-validation
2. Handling updates/corrections to collected information
3. Clarification of generated responses
"""

from langgraph.graph import StateGraph, START, END
from state import FamilyLawState
from nodes.query_analyzer import QueryAnalyzer
from nodes.information_gatherer import InformationGatherer
from nodes.retriever import retrieve_documents
from nodes.generator import generate_response
from node_logger import log_node_execution
from nodes.update_handler import preprocess_user_message
from typing import Dict
import logging

logger = logging.getLogger(__name__)


@log_node_execution("analyze_query")
def analyze_query_node(state: FamilyLawState) -> FamilyLawState:
    """
    Analyze query with support for re-validation.
    
    Now handles:
    1. Initial query analysis
    2. Post-gathering re-validation
    3. Update/correction scenarios
    """
    
    # Check if this is a re-validation after gathering
    is_revalidation = state.get("revalidation_mode", False)
    
    if is_revalidation:
        logger.info("🔄 RE-VALIDATING after information gathering")
        state["revalidation_mode"] = False  # Reset flag
    elif state.get("in_gathering_phase", False):
        logger.info("⏭️  Skipping analysis - already in gathering phase")
        return state
    elif state.get("analysis_complete", False) and not state.get("is_update", False):
        logger.info("⏭️  Skipping analysis - already complete")
        return state
    
    try:
        logger.info(f"🔍 === {'RE-' if is_revalidation else ''}ANALYZING QUERY ===")
        
        # Check if this is an update/correction scenario
        is_update = state.get("is_update", False)
        if is_update:
            logger.info("📝 Processing information update/correction")
        
        agent = QueryAnalyzer()
        response = agent.analyze_query(state)
        
        # Update state
        state["user_intent"] = response.get("user_intent")
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["has_sufficient_info"] = response.get("has_sufficient_info", False)
        
        # Merge new info with existing (for updates)
        new_info = response.get("info_collected", {})
        if is_update:
            existing_info = state.get("info_collected", {})
            existing_info.update(new_info)
            state["info_collected"] = existing_info
            logger.info(f"   Updated info: {list(state['info_collected'].keys())}")
        else:
            state["info_collected"] = new_info
        
        state["analysis_complete"] = True
        
        logger.info(f"   Intent: {state['user_intent']}")
        logger.info(f"   Info collected: {list(state['info_collected'].keys())}")
        logger.info(f"   Info needed: {state['info_needed_list']}")
        logger.info(f"   Sufficient: {state['has_sufficient_info']}")
        
        # Check intent confidence
        intent_confidence = response.get("intent_confidence", "high")
        if intent_confidence == "low" or not response.get("user_intent"):
            logger.info("❓ Low confidence - requesting clarification")
            state["needs_clarification"] = True
            state["clarification_question"] = "Could you please provide more details about your legal situation?"
        else:
            state["needs_clarification"] = False
            if not is_update and state["user_intent"]:
                state["root_query"] = state["query"]
            if not state["info_needed_list"]:
                logger.info("✅ No info needed - ready for retrieval")
                state["has_sufficient_info"] = True
                state["in_gathering_phase"] = False
            else:
                logger.info(f"📝 Need to gather {len(state['info_needed_list'])} items")
                state["in_gathering_phase"] = True
                state["gathering_step"] = 0
        
        # Reset update flag
        state["is_update"] = False
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Query Analyzer failed: {e}", exc_info=True)
        state["has_sufficient_info"] = True
        state["analysis_complete"] = True
        state["in_gathering_phase"] = False
        return state


@log_node_execution("gather_info")
def gather_information_node(state: FamilyLawState) -> FamilyLawState:
    """Gather information iteratively with logging."""
    
    try:
        step = state.get("gathering_step", 0)
        logger.info(f"📊 === GATHERING INFORMATION (Step {step}) ===")
        
        gatherer = InformationGatherer()
        response = gatherer.gather_next_information(state)
        
        # Update state
        state["info_collected"] = response.get("info_collected", {})
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["follow_up_question"] = response.get("follow_up_question")
        state["needs_more_info"] = response.get("needs_more_info", False)
        state["gathering_step"] = response.get("gathering_step", 0)
        state["current_question_target"] = response.get("current_question_target")
        
        logger.info(f"   ✓ Collected: {len(state['info_collected'])} items")
        logger.info(f"   ✓ Needed: {len(state['info_needed_list'])} items")
        
        # Check completion
        if not state["needs_more_info"]:
            logger.info("✅ Gathering complete - triggering re-validation")
            state["has_sufficient_info"] = True
            state["in_gathering_phase"] = False
            state["revalidation_mode"] = True  # Trigger re-validation
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Information Gatherer failed: {e}", exc_info=True)
        state["has_sufficient_info"] = True
        state["in_gathering_phase"] = False
        state["needs_more_info"] = False
        return state


@log_node_execution("revalidate")
def revalidate_information_node(state: FamilyLawState) -> FamilyLawState:
    """
    Re-validate collected information to ensure sufficiency.
    This runs after gathering is complete.
    """
    logger.info("🔄 === RE-VALIDATING COLLECTED INFORMATION ===")
    
    try:
        # Create a synthetic query that includes all collected info
        info_collected = state.get("info_collected", {})

        if state.get("revalidation_count", 0) >= 2 or len(info_collected) >= 10:
            logger.warning("⚠️  Maximum re-validation attempts reached, proceeding anyway")
            state["has_sufficient_info"] = True
            state["in_gathering_phase"] = False
            state["needs_more_info"] = False
            state["revalidation_mode"] = False
            return state
        
        original_query = state.get("root_query", "")
        
        # Format collected info as context
        info_context = "\n".join([
            f"- {key.replace('_', ' ').title()}: {value}"
            for key, value in info_collected.items()
        ])
        
        synthetic_query = f"{original_query}\n\nCollected Information:\n{info_context}"
        
        # Create temporary state for re-validation
        temp_state = dict(state)
        temp_state["query"] = synthetic_query
        temp_state["analysis_complete"] = False
        temp_state["in_gathering_phase"] = False
        
        # Run analyzer
        agent = QueryAnalyzer()
        response = agent.analyze_query(temp_state)
        
        # Check if we still need more info
        additional_info_needed = response.get("info_needed_list", [])
        
        if additional_info_needed:
            logger.info(f"⚠️  Re-validation found {len(additional_info_needed)} missing items")
            logger.info(f"   Additional info needed: {additional_info_needed}")
            
            # Add to existing needed list (avoid duplicates)
            current_needed = set(state.get("info_needed_list", []))
            current_needed.update(additional_info_needed)
            state["info_needed_list"] = list(current_needed)
            
            # Resume gathering
            state["in_gathering_phase"] = True
            state["needs_more_info"] = True
            state["has_sufficient_info"] = False
            state["revalidation_mode"] = False
            state["revalidation_count"] = state.get("revalidation_count", 0) + 1
            return state
        else:
            logger.info("✅ Re-validation passed - information is sufficient")
            state["has_sufficient_info"] = True
            state["in_gathering_phase"] = False
            state["needs_more_info"] = False
            state["revalidation_mode"] = False
            return state
    
    except Exception as e:
        logger.error(f"❌ Re-validation failed: {e}", exc_info=True)
        # On error, proceed anyway
        state["has_sufficient_info"] = True
        return state


@log_node_execution("retrieve")
def retrieve_documents_node(state: FamilyLawState) -> FamilyLawState:
    """Retrieve documents with logging."""
    result = retrieve_documents(state)
    state.update(result)
    return state


@log_node_execution("generate")
def generate_response_node(state: FamilyLawState) -> FamilyLawState:
    """Generate response with logging."""
    result = generate_response(state)
    state.update(result)
    return state


def route_after_analysis(state: FamilyLawState) -> str:
    """Route after initial query analysis."""
    if state.get("needs_clarification", False):
        logger.info("🔀 Routing → clarification")
        return "clarify"
    
    has_sufficient_info = state.get("has_sufficient_info", False)
    info_needed = state.get("info_needed_list", [])
    
    if has_sufficient_info or not info_needed:
        logger.info("🔀 Routing → retrieval (sufficient info)")
        return "retrieve"
    else:
        logger.info(f"🔀 Routing → gather_info (need {len(info_needed)} items)")
        return "gather_info"


def route_after_gathering(state: FamilyLawState) -> str:
    """Route after information gathering attempt."""
    needs_more_info = state.get("needs_more_info", False)
    revalidation_mode = state.get("revalidation_mode", False)
    
    if needs_more_info:
        logger.info("🔀 Routing → ask_question (more info needed)")
        return "ask_question"
    elif revalidation_mode:
        logger.info("🔀 Routing → revalidate (check sufficiency)")
        return "revalidate"
    else:
        logger.info("🔀 Routing → retrieval (gathering complete)")
        return "retrieve"


def route_after_revalidation(state: FamilyLawState) -> str:
    """Route after re-validation."""
    has_sufficient_info = state.get("has_sufficient_info", False)
    
    if has_sufficient_info:
        logger.info("🔀 Routing → retrieval (validation passed)")
        return "retrieve"
    else:
        logger.info("🔀 Routing → gather_info (need more info)")
        return "gather_info"


def format_clarification_response(state: FamilyLawState) -> dict:
    """Format clarification request."""
    clarification = state.get(
        "clarification_question",
        "Could you please clarify your legal situation?"
    )
    
    logger.info(f"❓ Sending clarification: {clarification[:100]}...")
    
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
    
    logger.info(f"📝 Asking follow-up: {follow_up[:100]}...")
    
    return {
        "response": follow_up,
        "sources": [],
        "message_type": "information_gathering",
        "info_collected": info_collected,
        "info_needed": info_needed
    }


def create_graph():
    """Create the enhanced family law assistant graph."""
    
    logger.info("🏗️  Building Enhanced LangGraph workflow...")
    
    workflow = StateGraph(FamilyLawState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("clarify", format_clarification_response)
    workflow.add_node("gather_info", gather_information_node)
    workflow.add_node("ask_question", format_follow_up_response)
    workflow.add_node("revalidate", revalidate_information_node)
    workflow.add_node("preprocess", preprocess_user_message)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("generate", generate_response_node)
    
    # Edges
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "analyze_query")
    
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
            "revalidate": "revalidate",
            "retrieve": "retrieve"
        }
    )
    
    workflow.add_edge("ask_question", END)
    
    # Re-validation routing
    workflow.add_conditional_edges(
        "revalidate",
        route_after_revalidation,
        {
            "retrieve": "retrieve",
            "gather_info": "gather_info"
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()
    logger.info("✅ Enhanced graph compiled successfully")
    
    return app


family_law_app = create_graph()