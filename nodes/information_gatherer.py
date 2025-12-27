"""
Enhanced Information Gatherer Node - Iteratively collects information.

This node:
1. Receives the list of needed information from query_analyzer
2. Asks ONE question at a time
3. Extracts the answer from user's response
4. Updates info_collected and info_needed_list
5. Continues until all information is gathered
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List
from state import FamilyLawState
import os
import json
import logging
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logger = logging.getLogger(__name__)


class InformationGatherer:
    """
    Gathers information iteratively through targeted questions.
    """
    
    QUESTION_GENERATION_PROMPT = """You are a compassionate Indian Family Law attorney conducting a client consultation.

SITUATION:
- User Intent: {user_intent}
- Information already collected: {info_collected}
- Information still needed: {info_needed}
- Current gathering step: {step}

YOUR TASK:
Generate ONE clear, empathetic, professional question to gather the MOST IMPORTANT missing information.

GUIDELINES:
1. Ask only ONE question at a time
2. Be empathetic and professional
3. Use simple, clear language
4. For domestic violence: prioritize safety questions
5. Make the question specific and easy to answer
6. Refer to previously collected information naturally

Format your response as JSON:
{{
  "question": "your question here",
  "info_target": "the specific information key this question addresses"
}}

YOUR QUESTION (JSON only):"""

    ANSWER_EXTRACTION_PROMPT = """You are an information extraction expert. Extract ONLY the relevant answer from the user's response.

QUESTION CONTEXT: We asked about {info_target}
USER'S ANSWER: "{user_response}"

Extract the direct answer. Be concise and specific.

Return JSON:
{{
  "extracted_value": "the answer only, not the full sentence"
}}

Examples:
- If user says "I got married in 2015" → "2015"
- If user says "yes, currently married" → "currently married"  
- If user says "We have 2 children, ages 5 and 8" → "2 children, ages 5 and 8"

EXTRACTION (JSON only):"""
    
    def __init__(self, huggingface_api_key: str = None):
        """Initialize the InformationGatherer with LLM."""
        api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=api_key,
                task="text-generation",
                max_new_tokens=512,
            )
        )
    
    def gather_next_information(self, state: FamilyLawState) -> Dict:
        """
        Generate next question or extract information from user's response.
        
        Returns:
            Dict with:
            - needs_more_info: bool
            - follow_up_question: str (if needs_more_info is True)
            - info_collected: dict
            - info_needed_list: list
        """
        query = state["query"]
        messages = state.get("messages", [])
        info_collected = dict(state.get("info_collected", {}))  # Create a copy
        info_needed_list = list(state.get("info_needed_list", []))  # Create a copy
        user_intent = state.get("user_intent", "legal advice")
        gathering_step = state.get("gathering_step", 0)
        
        logger.info(f"Gathering step {gathering_step}, needed: {info_needed_list}, collected: {list(info_collected.keys())}")
        
        # If no information is needed, we're done
        if not info_needed_list:
            logger.info("No information needed, gathering complete")
            return {
                "needs_more_info": False,
                "info_collected": info_collected,
                "info_needed_list": [],
                "gathering_step": gathering_step
            }
        
        # Check if we're processing a user response (gathering_step > 0 means we already asked a question)
        if gathering_step > 0:
            # Find the last user message (the answer to our question)
            last_user_message = None
            
            # Look for the most recent HumanMessage that isn't the initial query
            for msg in reversed(messages):
                if msg.__class__.__name__ == "HumanMessage":
                    # Skip if this is the first message (initial query)
                    if len([m for m in messages if m.__class__.__name__ == "HumanMessage"]) > gathering_step:
                        last_user_message = msg
                        break
            
            if last_user_message and info_needed_list:
                # Extract the answer for the current target
                current_target = info_needed_list[0]
                logger.info(f"Extracting answer for: {current_target}")
                logger.info(f"User response: {last_user_message.content[:100]}")
                
                extracted_info = self._extract_information(
                    last_user_message.content,
                    current_target
                )
                
                logger.info(f"Extraction result: confidence={extracted_info['confidence']}, value={extracted_info['extracted_value'][:50]}")
                
                # Always store the answer (even if confidence is low, we move forward)
                info_collected[current_target] = extracted_info["extracted_value"]
                info_needed_list = info_needed_list[1:]  # Remove this item
                
                logger.info(f"✓ Stored {current_target}")
                logger.info(f"Remaining items: {info_needed_list}")
            else:
                logger.warning(f"No user message found or no items needed")
        
        # Check if we still need more information
        if not info_needed_list:
            logger.info("All information collected successfully!")
            return {
                "needs_more_info": False,
                "info_collected": info_collected,
                "info_needed_list": [],
                "gathering_step": gathering_step
            }
        
        # Generate next question
        next_question = self._generate_question(
            user_intent,
            info_collected,
            info_needed_list,
            gathering_step
        )
        
        logger.info(f"Generated question for next item: {info_needed_list[0]}")
        
        return {
            "needs_more_info": True,
            "follow_up_question": next_question,
            "info_collected": info_collected,
            "info_needed_list": info_needed_list,
            "gathering_step": gathering_step + 1
        }
    
    def _generate_question(
        self,
        user_intent: str,
        info_collected: Dict,
        info_needed_list: List[str],
        step: int
    ) -> str:
        """Generate a question for the next piece of needed information."""
        
        # Format collected and needed info
        collected_str = self._format_info_collected(info_collected)
        needed_str = self._format_info_needed(info_needed_list)
        
        # Prepare prompt
        prompt = self.QUESTION_GENERATION_PROMPT.format(
            user_intent=user_intent.replace("_", " ").title(),
            info_collected=collected_str,
            info_needed=needed_str,
            step=step
        )
        
        try:
            conversation = [
                SystemMessage(content="You are an empathetic family law attorney. Respond ONLY with JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(conversation)
            response_text = response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            question_data = json.loads(response_text)
            question = question_data.get("question", "")
            
            # Clean up the question
            if ":" in question and len(question.split(":")[0].split()) <= 3:
                question = question.split(":", 1)[1].strip()
            
            return question
        
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            # Fallback to simple question
            next_needed = info_needed_list[0].replace("_", " ").title()
            return f"Could you please provide details about: {next_needed}?"
    
    def _extract_information(self, user_response: str, info_target: str) -> Dict:
        """Extract information from user's response."""
        
        prompt = self.ANSWER_EXTRACTION_PROMPT.format(
            info_target=info_target.replace("_", " ").title(),
            user_response=user_response
        )
        
        try:
            conversation = [
                SystemMessage(content="You extract answers from text. Respond ONLY with JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(conversation)
            response_text = response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            extraction = json.loads(response_text)
            
            extracted_value = extraction.get("extracted_value", user_response).strip()
            
            # If extraction is empty or just says "not provided", use full response
            if not extracted_value or extracted_value.lower() in ["not provided", "none", "n/a"]:
                extracted_value = user_response.strip()
            
            logger.info(f"Extracted: {extracted_value[:100]}")
            
            return {
                "extracted_value": extracted_value,
                "confidence": "high"  # Always proceed
            }
        
        except Exception as e:
            logger.error(f"Error extracting information: {str(e)}")
            # Fallback: use the full response
            return {
                "extracted_value": user_response.strip(),
                "confidence": "high"  # Always proceed to avoid loops
            }
    
    def _format_info_collected(self, info_collected: Dict) -> str:
        """Format collected information for display."""
        if not info_collected:
            return "No information collected yet."
        
        formatted = []
        for key, value in info_collected.items():
            formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        return "\n".join(formatted)
    
    def _format_info_needed(self, info_needed_list: List[str]) -> str:
        """Format the list of needed information."""
        if not info_needed_list:
            return "All information collected."
        
        formatted = []
        for i, item in enumerate(info_needed_list, 1):
            formatted.append(f"{i}. {item.replace('_', ' ').title()}")
        return "\n".join(formatted)