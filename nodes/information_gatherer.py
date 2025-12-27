"""
Fixed Information Gatherer Node - Properly updates state between iterations.

Key fixes:
1. Correctly identifies when processing user responses vs asking new questions
2. Properly updates info_collected and removes from info_needed_list
3. Better extraction logic
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
- Next information needed: {current_target}
- All remaining needs: {all_remaining}

YOUR TASK:
Ask ONE clear, empathetic question to gather information about: {current_target}

GUIDELINES:
1. Ask only about {current_target} - be specific
2. Be empathetic and professional
3. Use simple, clear language
4. Reference previously collected information naturally if relevant
5. Make it easy for the user to answer

YOUR QUESTION:"""

    ANSWER_EXTRACTION_PROMPT = """Extract the answer from the user's response.

QUESTION ASKED: We need information about "{info_target}"
USER'S RESPONSE: {user_response}

Extract ONLY the relevant information that answers what we asked about.
Be concise but capture all relevant details.

If the user didn't provide the information, respond with "NOT_PROVIDED"

EXTRACTED ANSWER:"""
    
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
        Main logic: Extract answer from previous response OR ask next question.
        
        Flow:
        1. If gathering_step > 0: Extract answer from last user message
        2. Update info_collected, remove from info_needed_list
        3. If still need info: Generate next question
        4. If no more info needed: Mark as complete
        """
        query = state["query"]
        messages = state.get("messages", [])
        info_collected = dict(state.get("info_collected", {}))
        info_needed_list = list(state.get("info_needed_list", []))
        user_intent = state.get("user_intent", "legal advice")
        gathering_step = state.get("gathering_step", 0)
        
        logger.info(f"=== Gathering Step {gathering_step} ===")
        logger.info(f"Info needed: {info_needed_list}")
        logger.info(f"Info collected keys: {list(info_collected.keys())}")
        
        # STEP 1: Check if we need to extract an answer from previous response
        if gathering_step > 0 and info_needed_list:
            # Get the target we were asking about (stored separately or use first in list)
            current_target = state.get("current_question_target") or info_needed_list[0]
            
            # Find the last user message
            last_user_msg = None
            user_messages = [m for m in messages if m.__class__.__name__ == "HumanMessage"]
            
            # Skip the initial query, get the response to our question
            if len(user_messages) > gathering_step:
                last_user_msg = user_messages[gathering_step]
                
            if last_user_msg:
                logger.info(f"Extracting answer for: {current_target}")
                logger.info(f"From response: {last_user_msg.content[:100]}...")
                
                # Extract the information
                extracted = self._extract_information(
                    last_user_msg.content,
                    current_target
                )
                
                logger.info(f"Extracted value: {extracted[:100]}...")
                
                # CRITICAL: Store the answer and remove from needed list
                if extracted and extracted != "NOT_PROVIDED":
                    info_collected[current_target] = extracted
                    # Remove this item from needed list
                    if current_target in info_needed_list:
                        info_needed_list.remove(current_target)
                    
                    logger.info(f"✓ Stored answer for: {current_target}")
                    logger.info(f"Updated needed list: {info_needed_list}")
                else:
                    logger.warning(f"User didn't provide {current_target}, keeping in list")
        
        # STEP 2: Check if we're done
        if not info_needed_list:
            logger.info("✓ All information collected! Gathering complete.")
            return {
                "needs_more_info": False,
                "info_collected": info_collected,
                "info_needed_list": [],
                "gathering_step": gathering_step,
                "current_question_target": None
            }
        
        # STEP 3: Generate next question for the first item in needed list
        next_target = info_needed_list[0]
        logger.info(f"Generating question for: {next_target}")
        
        next_question = self._generate_question(
            user_intent,
            info_collected,
            next_target,
            info_needed_list
        )
        
        logger.info(f"Generated question: {next_question[:100]}...")
        
        return {
            "needs_more_info": True,
            "follow_up_question": next_question,
            "info_collected": info_collected,
            "info_needed_list": info_needed_list,
            "gathering_step": gathering_step + 1,
            "current_question_target": next_target  # Store what we're asking about
        }
    
    def _generate_question(
        self,
        user_intent: str,
        info_collected: Dict,
        current_target: str,
        all_remaining: List[str]
    ) -> str:
        """Generate a focused question for the specific information needed."""
        
        collected_str = self._format_info_collected(info_collected)
        
        # Prepare prompt
        prompt = self.QUESTION_GENERATION_PROMPT.format(
            user_intent=user_intent.replace("_", " ").title(),
            info_collected=collected_str or "None yet",
            current_target=current_target.replace("_", " ").title(),
            all_remaining=", ".join([x.replace("_", " ") for x in all_remaining])
        )
        
        try:
            conversation = [
                SystemMessage(content="You are an empathetic family law attorney. Ask ONE focused question."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(conversation)
            question = response.content.strip()
            
            # Clean up any extra text
            if "YOUR QUESTION:" in question:
                question = question.split("YOUR QUESTION:")[-1].strip()
            if "Question:" in question:
                question = question.split("Question:")[-1].strip()
            
            # Remove quotes if present
            question = question.strip('"\'')
            
            return question
        
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            # Fallback
            return f"Could you please provide information about: {current_target.replace('_', ' ')}?"
    
    def _extract_information(self, user_response: str, info_target: str) -> str:
        """Extract specific information from user's response."""
        
        prompt = self.ANSWER_EXTRACTION_PROMPT.format(
            info_target=info_target.replace("_", " ").title(),
            user_response=user_response
        )
        
        try:
            conversation = [
                SystemMessage(content="Extract only the relevant answer. Be concise."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(conversation)
            extracted = response.content.strip()
            
            # Clean up
            if "EXTRACTED ANSWER:" in extracted:
                extracted = extracted.split("EXTRACTED ANSWER:")[-1].strip()
            
            extracted = extracted.strip('"\'')
            
            # If extraction failed or empty, use full response
            if not extracted or len(extracted) < 3:
                extracted = user_response.strip()
            
            return extracted
        
        except Exception as e:
            logger.error(f"Error extracting information: {str(e)}")
            return user_response.strip()
    
    def _format_info_collected(self, info_collected: Dict) -> str:
        """Format collected information for display."""
        if not info_collected:
            return "No information collected yet."
        
        formatted = []
        for key, value in info_collected.items():
            formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        return "\n".join(formatted)