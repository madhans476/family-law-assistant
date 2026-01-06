"""
Smart update and correction handler for handling follow-up interactions.

Handles scenarios like:
1. User provides additional information after initial response
2. User corrects previously provided information
3. User asks clarifying questions about the generated response
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Optional, Literal
import os
import logging
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logger = logging.getLogger(__name__)


class UpdateHandler:
    """
    Intelligently handles updates, corrections, and follow-up queries.
    """
    
    INTENT_CLASSIFICATION_PROMPT = """Analyze the user's message in the context of an ongoing legal consultation.

CONVERSATION CONTEXT:
Previous Response Generated: {has_previous_response}
Information Already Collected: {info_collected}

USER'S NEW MESSAGE:
{user_message}

CLASSIFY THE USER'S INTENT INTO ONE OF THESE CATEGORIES:

1. **NEW_INFO_ADDITION**: User is providing additional information they forgot to mention
   Example: "Oh, I also forgot to mention we have two children"
   
2. **CORRECTION**: User is correcting previously provided information
   Example: "Actually, we got married in 2020, not 2019"
   
3. **CLARIFICATION_REQUEST**: User wants clarification about the response given
   Example: "What does Section 498A mean?", "Can you explain more about the procedure?"
   
4. **NEW_QUESTION**: User has a completely new legal question
   Example: "Now I want to ask about property division"
   
5. **DOUBT_ABOUT_RESPONSE**: User doubts or challenges the advice given
   Example: "Are you sure about this?", "My friend's lawyer said something different"

RESPOND WITH JSON:
{{
  "intent_type": "new_info_addition|correction|clarification_request|new_question|doubt_about_response",
  "confidence": "high|medium|low",
  "specific_topic": "what the user is referring to",
  "requires_reprocessing": true|false
}}

Rules:
- requires_reprocessing=true if we need to regenerate advice (new info, corrections)
- requires_reprocessing=false if just clarification needed
- Be conservative: when unsure, classify as clarification_request

YOUR CLASSIFICATION:"""
    
    def __init__(self, huggingface_api_key: str = None):
        """Initialize the update handler."""
        api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=api_key,
                task="text-generation",
                max_new_tokens=512,
            )
        )
    
    def classify_followup_intent(
        self,
        user_message: str,
        has_previous_response: bool,
        info_collected: Dict[str, str]
    ) -> Dict:
        """
        Classify what the user intends with their follow-up message.
        
        Returns:
            Dict with intent classification and processing instructions
        """
        logger.info("🔍 === CLASSIFYING FOLLOW-UP INTENT ===")
        
        try:
            # Format collected info
            info_str = "\n".join([
                f"- {k.replace('_', ' ').title()}: {v}"
                for k, v in info_collected.items()
            ]) if info_collected else "None"
            
            prompt = self.INTENT_CLASSIFICATION_PROMPT.format(
                has_previous_response=str(has_previous_response),
                info_collected=info_str,
                user_message=user_message
            )
            
            conversation = [
                SystemMessage(content="You are an intent classifier. Respond ONLY with valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(conversation)
            response_text = response.content.strip()
            
            # Extract JSON
            import json
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            classification = json.loads(response_text)
            
            intent_type = classification.get("intent_type", "clarification_request")
            requires_reprocessing = classification.get("requires_reprocessing", False)
            
            logger.info(f"   Intent: {intent_type}")
            logger.info(f"   Reprocess: {requires_reprocessing}")
            
            return classification
        
        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}", exc_info=True)
            # Safe fallback
            return {
                "intent_type": "clarification_request",
                "confidence": "low",
                "specific_topic": "unknown",
                "requires_reprocessing": False
            }
    
    def handle_update(self, state: Dict) -> Dict:
        """
        Main entry point for handling follow-up interactions.
        
        This determines:
        1. What the user intends
        2. Whether we need to reprocess (re-analyze, re-gather, re-generate)
        3. How to route the request
        """
        query = state.get("query", "")
        messages = state.get("messages", [])
        info_collected = state.get("info_collected", {})
        response = state.get("response", "")
        
        # Check if this is a follow-up (we have previous messages)
        has_previous_response = len(messages) > 2 and bool(response)
        
        if not has_previous_response:
            # First interaction, proceed normally
            logger.info("First interaction - no update handling needed")
            return state
        
        # Classify the intent
        classification = self.classify_followup_intent(
            user_message=query,
            has_previous_response=has_previous_response,
            info_collected=info_collected
        )
        
        intent_type = classification["intent_type"]
        requires_reprocessing = classification["requires_reprocessing"]
        
        # Update state based on intent
        if intent_type == "new_info_addition":
            logger.info("📝 Handling: New information addition")
            state["is_update"] = True
            state["update_type"] = "addition"
            state["session_phase"] = "updating"
            # Will trigger re-analysis to extract new info
            state["analysis_complete"] = False
            
        elif intent_type == "correction":
            logger.info("✏️ Handling: Information correction")
            state["is_update"] = True
            state["update_type"] = "correction"
            state["session_phase"] = "updating"
            # Will trigger re-analysis to update info
            state["analysis_complete"] = False
            
        elif intent_type == "clarification_request":
            logger.info("❓ Handling: Clarification request")
            state["is_update"] = False
            state["update_type"] = "clarification"
            # Generate clarification without full reprocessing
            state["needs_clarification"] = True
            state["clarification_question"] = self._generate_clarification_response(
                query, response, info_collected
            )
            
        elif intent_type == "doubt_about_response":
            logger.info("🤔 Handling: Doubt about response")
            state["is_update"] = False
            state["update_type"] = "clarification"
            # Address doubt with additional explanation
            state["needs_clarification"] = True
            state["clarification_question"] = self._address_doubt(
                query, response, info_collected
            )
            
        elif intent_type == "new_question":
            logger.info("🆕 Handling: New question")
            # Treat as completely new conversation
            state["is_update"] = False
            state["analysis_complete"] = False
            state["session_phase"] = "initial"
            # Keep collected info but re-analyze
        
        return state
    
    def _generate_clarification_response(
        self,
        query: str,
        previous_response: str,
        info_collected: Dict
    ) -> str:
        """Generate clarification response without full reprocessing."""
        
        CLARIFICATION_PROMPT = f"""The user asked for clarification about your previous legal advice.

PREVIOUS ADVICE:
{previous_response[:1000]}

USER'S CLARIFICATION REQUEST:
{query}

Provide a clear, focused clarification that:
1. Directly answers their specific question
2. References the relevant part of your previous advice
3. Adds any necessary additional context
4. Uses simple language

Keep it concise.

YOUR CLARIFICATION:"""
        
        try:
            conversation = [
                SystemMessage(content="You are a helpful legal assistant providing clarifications."),
                HumanMessage(content=CLARIFICATION_PROMPT)
            ]
            
            response = self.llm.invoke(conversation)
            return response.content.strip()
        
        except Exception as e:
            logger.error(f"Failed to generate clarification: {e}")
            return f"I understand you need clarification about: {query}. Let me provide more detail on this specific point..."
    
    def _address_doubt(
        self,
        query: str,
        previous_response: str,
        info_collected: Dict
    ) -> str:
        """Address user's doubt or concern about the advice."""
        
        DOUBT_PROMPT = f"""The user has expressed doubt or concern about your legal advice.

PREVIOUS ADVICE:
{previous_response[:1000]}

USER'S CONCERN:
{query}

Respond in a way that:
1. Acknowledges their concern respectfully
2. Provides additional explanation or reasoning
3. Cites relevant laws or precedents if applicable
4. Encourages them to verify with a lawyer if needed
5. Maintains professional confidence while being understanding

YOUR RESPONSE:"""
        
        try:
            conversation = [
                SystemMessage(content="You are a professional legal assistant addressing concerns."),
                HumanMessage(content=DOUBT_PROMPT)
            ]
            
            response = self.llm.invoke(conversation)
            return response.content.strip()
        
        except Exception as e:
            logger.error(f"Failed to address doubt: {e}")
            return "I understand your concern. Let me provide additional context on this matter..."


# Integration with main.py
def preprocess_user_message(state: Dict) -> Dict:
    """
    Add this as the first node in your graph to handle updates.
    
    Usage in graph.py:
        workflow.add_node("preprocess", preprocess_user_message)
        workflow.add_edge(START, "preprocess")
        workflow.add_edge("preprocess", "analyze_query")
    """
    handler = UpdateHandler()
    updated_state = handler.handle_update(state)
    return updated_state