"""
graph.py with proper logging configuration.

Make sure logging is configured in main.py BEFORE importing this module.
"""

from langgraph.graph import StateGraph, START, END
from state import FamilyLawState
from nodes.query_analyzer import QueryAnalyzer
from nodes.information_gatherer import InformationGatherer
from nodes.retriever import retrieve_documents
from nodes.generator import generate_response
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


def analyze_query_node(state: FamilyLawState) -> FamilyLawState:
    """Analyze ONLY the initial query. Skip if in gathering phase."""
    
    # CRITICAL: Don't analyze follow-up responses
    if state.get("in_gathering_phase", False):
        logger.info("⏭️  Skipping analysis - already in gathering phase")
        return state
    
    if state.get("analysis_complete", False):
        logger.info("⏭️  Skipping analysis - already complete")
        return state
    
    try:
        logger.info("🔍 === ANALYZING INITIAL QUERY ===")
        agent = QueryAnalyzer()
        response = agent.analyze_query(state)
        
        # Update state
        state["user_intent"] = response.get("user_intent")
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["has_sufficient_info"] = response.get("has_sufficient_info", False)
        state["info_collected"] = response.get("info_collected", {})
        state["analysis_complete"] = True
        
        logger.info(f"   Intent: {state['user_intent']}")
        logger.info(f"   Info collected: {list(state['info_collected'].keys())}")
        logger.info(f"   Info needed: {state['info_needed_list']}")
        
        # Check intent confidence
        intent_confidence = response.get("intent_confidence", "high")
        if intent_confidence == "low" or not response.get("user_intent"):
            logger.info("❓ Low confidence - requesting clarification")
            state["needs_clarification"] = True
            state["clarification_question"] = "Could you please provide more details about your legal situation?"
        else:
            state["needs_clarification"] = False
            
            if not state["info_needed_list"]:
                logger.info("✅ No info needed - ready for retrieval")
                state["has_sufficient_info"] = True
                state["in_gathering_phase"] = False
            else:
                logger.info(f"📝 Need to gather {len(state['info_needed_list'])} items")
                state["in_gathering_phase"] = True
                state["gathering_step"] = 0
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Query Analyzer failed: {e}", exc_info=True)
        state["has_sufficient_info"] = True
        state["analysis_complete"] = True
        state["in_gathering_phase"] = False
        return state


def gather_information_node(state: FamilyLawState) -> FamilyLawState:
    """Gather information iteratively."""
    
    try:
        step = state.get("gathering_step", 0)
        logger.info(f"📊 === GATHERING INFORMATION (Step {step}) ===")
        logger.info(f"   Currently collected: {list(state.get('info_collected', {}).keys())}")
        logger.info(f"   Still needed: {state.get('info_needed_list', [])}")
        logger.info(f"   Messages count: {len(state.get('messages', []))}")
        
        gatherer = InformationGatherer()
        response = gatherer.gather_next_information(state)
        
        # CRITICAL: Update state with new values
        old_collected = len(state.get("info_collected", {}))
        old_needed = len(state.get("info_needed_list", []))
        
        state["info_collected"] = response.get("info_collected", {})
        state["info_needed_list"] = response.get("info_needed_list", [])
        state["follow_up_question"] = response.get("follow_up_question")
        state["needs_more_info"] = response.get("needs_more_info", False)
        state["gathering_step"] = response.get("gathering_step", 0)
        state["current_question_target"] = response.get("current_question_target")
        
        new_collected = len(state["info_collected"])
        new_needed = len(state["info_needed_list"])
        
        logger.info(f"   ✓ Collected: {old_collected} → {new_collected}")
        logger.info(f"   ✓ Needed: {old_needed} → {new_needed}")
        logger.info(f"   ✓ Current target: {state['current_question_target']}")
        
        # Check completion
        if not state["needs_more_info"]:
            logger.info("✅ Gathering complete!")
            state["has_sufficient_info"] = True
            state["in_gathering_phase"] = False
        else:
            logger.info(f"➡️  Asking next question about: {state['current_question_target']}")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Information Gatherer failed: {e}", exc_info=True)
        state["has_sufficient_info"] = True
        state["in_gathering_phase"] = False
        state["needs_more_info"] = False
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
    
    if needs_more_info:
        logger.info("🔀 Routing → ask_question (more info needed)")
        return "ask_question"
    else:
        logger.info("🔀 Routing → retrieval (gathering complete)")
        return "retrieve"


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
    logger.info(f"   Progress: {len(info_collected)} collected, {len(info_needed)} remaining")
    
    return {
        "response": follow_up,
        "sources": [],
        "message_type": "information_gathering",
        "info_collected": info_collected,
        "info_needed": info_needed
    }


def create_graph():
    """Create the family law assistant graph."""
    
    logger.info("🏗️  Building LangGraph workflow...")
    
    workflow = StateGraph(FamilyLawState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("clarify", format_clarification_response)
    workflow.add_node("gather_info", gather_information_node)
    workflow.add_node("ask_question", format_follow_up_response)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_response)
    
    # Edges
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
    logger.info("✅ Graph compiled successfully")
    
    return app


family_law_app = create_graph()