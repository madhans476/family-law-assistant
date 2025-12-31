"""
Debug logging helper to track state changes and identify issues.

Add this to your nodes to see what's happening.
"""

import logging
import json

logger = logging.getLogger(__name__)


def log_state_transition(node_name: str, state_before: dict, state_after: dict):
    """
    Log state changes between node executions.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"NODE: {node_name}")
    logger.info(f"{'='*60}")
    
    # Log gathering phase info
    logger.info(f"IN_GATHERING_PHASE: {state_before.get('in_gathering_phase')} → {state_after.get('in_gathering_phase')}")
    logger.info(f"GATHERING_STEP: {state_before.get('gathering_step')} → {state_after.get('gathering_step')}")
    logger.info(f"HAS_SUFFICIENT_INFO: {state_before.get('has_sufficient_info')} → {state_after.get('has_sufficient_info')}")
    
    # Log info collection
    before_collected = list(state_before.get('info_collected', {}).keys())
    after_collected = list(state_after.get('info_collected', {}).keys())
    logger.info(f"INFO_COLLECTED: {before_collected} → {after_collected}")
    
    before_needed = state_before.get('info_needed_list', [])
    after_needed = state_after.get('info_needed_list', [])
    logger.info(f"INFO_NEEDED: {before_needed} → {after_needed}")
    
    # Log current target
    logger.info(f"CURRENT_TARGET: {state_before.get('current_question_target')} → {state_after.get('current_question_target')}")
    
    # Log message count
    before_msgs = len(state_before.get('messages', []))
    after_msgs = len(state_after.get('messages', []))
    logger.info(f"MESSAGES: {before_msgs} → {after_msgs}")
    
    logger.info(f"{'='*60}\n")


def log_gathering_iteration(step: int, state: dict, action: str):
    """
    Log details of each gathering iteration.
    """
    logger.info(f"\n{'*'*40}")
    logger.info(f"GATHERING ITERATION {step}: {action}")
    logger.info(f"{'*'*40}")
    logger.info(f"Info Collected: {json.dumps(state.get('info_collected', {}), indent=2)}")
    logger.info(f"Info Needed: {state.get('info_needed_list', [])}")
    logger.info(f"Current Target: {state.get('current_question_target')}")
    logger.info(f"Messages Count: {len(state.get('messages', []))}")
    
    # Log last few messages
    messages = state.get('messages', [])
    if messages:
        logger.info(f"\nLast 3 messages:")
        for msg in messages[-3:]:
            role = msg.__class__.__name__
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            logger.info(f"  {role}: {content}")
    
    logger.info(f"{'*'*40}\n")
