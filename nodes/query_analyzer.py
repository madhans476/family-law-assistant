"""
Query Analyzer Node - Analyzes user queries to understand intent and information needs.

This node identifies:
1. Case type (divorce, domestic violence, custody, etc.)
2. Information already provided by the user
3. Critical information still needed
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List
from state import FamilyLawState
import os
import json
import logging
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes user queries to understand intent, case type, and information needs.
    """
    
    QUERY_ANALYSIS_PROMPT = """You are an expert Indian FAMILY LAW analyst. Analyze the user's query to understand their legal situation.

Your task is to:
1. Identify the PRIMARY case type or intent, it can be divorce, domestic_violence, dowry, child_custody, maintenance, general, etc.,
2. Extract what information the user HAS PROVIDED.
3. Identify what CRITICAL information is STILL NEEDED to give proper legal advice, no assumptions.

RESPONSE FORMAT (Strictly JSON only, no other text):
{{
  "user_intent": "brief description of what user wants to achieve",
  "info_provided": {{
    "key1": "value1",
    "key2": "value2"
  }},
  "info_needed": [
    "specific_info_1",
    "specific_info_2"
  ],
}}

RULES:
- If user provides comprehensive details, set "info_needed": []
- If critical information is missing, list specific needs
- Extract dates, relationships, incidents from the query
- Be specific about what's needed (e.g., "marriage_date" not just "details")

USER QUERY:
{query}

ANALYSIS (JSON only):"""
    
    def __init__(self, huggingface_api_key: str = None):
        """
        Initialize the QueryAnalyzer with LLM.
        
        Args:
            huggingface_api_key: HuggingFace API key (optional, uses env var if not provided)
        """
        api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=api_key,
                task="text-generation",
            )
        )
    
    def analyze_query(self, state: FamilyLawState) -> Dict:
        """
        Analyze the user's query to understand intent and information needs.
        
        Args:
            state: FamilyLawState containing the query
            
        Returns:
            Dict with:
            - info_collected: dict
            - info_needed_list: list
            - has_sufficient_info: bool
            - user_intent: str
        """
        query = state["query"]
        
        # Prepare prompt
        prompt = self.QUERY_ANALYSIS_PROMPT.format(query=query)
        
        try:
            # Get LLM analysis
            conversation = [
                SystemMessage(content="You are a legal query analyzer. Respond ONLY with valid JSON."),
                HumanMessage(content=prompt)
            ]
            logger.info("Invoking LLM for query analysis")
            response = self.llm.invoke(conversation)
            response_text = response.content.strip()
            
            # Extract JSON from response
            # Sometimes LLM adds markdown backticks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            analysis = json.loads(response_text)
            
            # Validate and set defaults
            info_provided = analysis.get("info_provided", {})
            info_needed = analysis.get("info_needed", [])
            has_sufficient_info = analysis.get("has_sufficient_info", False)
            user_intent = analysis.get("user_intent", "Seeking legal advice")
            
            logger.info(f"Query analysis: sufficient_info={has_sufficient_info}, needs={len(info_needed)} items")
            
            return {
                "info_collected": info_provided,
                "info_needed_list": info_needed,
                "has_sufficient_info": has_sufficient_info,
                "user_intent": user_intent
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}\nResponse: {response_text[:200]}")
            # Fallback: basic keyword analysis
            return self.fallback_analysis(query)
        
        except Exception as e:
            logger.error(f"Query analysis error: {str(e)}")
            return self.fallback_analysis(query)
    
    def fallback_analysis(self, query: str) -> Dict:
        """
        Fallback keyword-based analysis if LLM fails.
        
        Args:
            query: User query string
            
        Returns:
            Dict with analysis results
        """
        query_lower = query.lower()
        
        # Determine case type
        if any(word in query_lower for word in ["violence", "abuse", "beat", "assault", "hit", "threat"]):
            case_type = "domestic_violence"
        elif any(word in query_lower for word in ["dowry", "dahej", "demand", "harassment"]):
            case_type = "dowry"
        elif any(word in query_lower for word in ["custody", "children", "child", "visitation"]):
            case_type = "child_custody"
        elif any(word in query_lower for word in ["divorce", "separation", "marriage"]):
            case_type = "divorce"
        elif any(word in query_lower for word in ["maintenance", "alimony", "support"]):
            case_type = "maintenance"
        else:
            case_type = "general"
        
        # Basic information extraction
        info_provided = {}
        if "married" in query_lower or "marriage" in query_lower:
            info_provided["marriage_mentioned"] = "yes"
        if "child" in query_lower:
            info_provided["children_mentioned"] = "yes"
        
        # Determine if we need more info
        has_sufficient_info = len(query.split()) > 30  # If query is detailed
        
        info_needed = []
        if not has_sufficient_info:
            if case_type == "divorce":
                info_needed = ["marriage_date", "grounds_for_divorce", "children_details"]
            elif case_type == "domestic_violence":
                info_needed = ["current_safety_status", "incident_details", "relationship_to_perpetrator"]
            elif case_type == "child_custody":
                info_needed = ["children_ages", "current_custody_arrangement"]
            else:
                info_needed = ["detailed_situation", "timeline_of_events"]
        
        logger.info(f"Fallback analysis: case_type={case_type}")
        
        return {
            "case_type": case_type,
            "info_collected": info_provided,
            "info_needed_list": info_needed,
            "has_sufficient_info": has_sufficient_info,
            "user_intent": "Seeking legal advice"
        }



# """
# Query Analyzer Node - Analyzes user queries to understand intent and information needs.

# This node identifies:
# 1. Case type (divorce, domestic violence, custody, etc.)
# 2. Information already provided by the user
# 3. Critical information still needed
# """

# from langchain_core.messages import HumanMessage, SystemMessage
# from typing import Dict, List
# from state import FamilyLawState
# import os
# import json
# import logging
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# logger = logging.getLogger(__name__)

# # Initialize LLM
# llm = ChatHuggingFace(
#     llm=HuggingFaceEndpoint(
#         repo_id="meta-llama/Llama-3.1-8B-Instruct",
#         huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
#         task="text-generation",
#     )
# )

# QUERY_ANALYSIS_PROMPT = """You are an expert Indian FAMILY LAW analyst. Analyze the user's query to understand their legal situation.

# Your task is to:
# 1. Identify the PRIMARY case type or intent, it can be divorce, domestic_violence, dowry, child_custody, maintenance, general, etc.,
# 2. Extract what information the user HAS PROVIDED.
# 3. Identify what CRITICAL information is STILL NEEDED to give proper legal advice, no assumptions.

# RESPONSE FORMAT (Strictly JSON only, no other text):
# {{
#   "user_intent": "brief description of what user wants to achieve",
#   "info_provided": {{
#     "key1": "value1",
#     "key2": "value2"
#   }},
#   "info_needed": [
#     "specific_info_1",
#     "specific_info_2"
#   ],
#   "has_sufficient_info": true/false
# }}

# RULES:
# - If user provides comprehensive details, set "has_sufficient_info": true and "info_needed": []
# - If critical information is missing, set "has_sufficient_info": false and list specific needs
# - Extract dates, relationships, incidents from the query
# - Be specific about what's needed (e.g., "marriage_date" not just "details")

# USER QUERY:
# {query}

# ANALYSIS (JSON only):"""

# def analyze_query(state: FamilyLawState) -> Dict:
#     """
#     Analyze the user's query to understand intent and information needs.
    
#     Returns:
#         Dict with:
#         - case_type: str
#         - info_provided: dict
#         - info_needed: list
#         - has_sufficient_info: bool
#         - user_intent: str
#     """
#     query = state["query"]
#     # messages = state.get("messages", [])
    
#     # Format conversation history
#     history = ""
#     # if messages:
#     #     recent_messages = messages[-4:]  # Last 2 exchanges
#     #     history_lines = []
#     #     for msg in recent_messages:
#     #         role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
#     #         history_lines.append(f"{role}: {msg.content[:150]}")
#     #     history = "\n".join(history_lines)
#     # else:
#     #     history = "No previous conversation."
    
#     # Prepare prompt
#     prompt = QUERY_ANALYSIS_PROMPT.format(
#         query=query,
#     )
    
#     try:
#         # Get LLM analysis
#         conversation = [
#             SystemMessage(content="You are a legal query analyzer. Respond ONLY with valid JSON."),
#             HumanMessage(content=prompt)
#         ]
#         logger.info("Invoking LLM for query analysis")
#         response = llm.invoke(conversation)
#         response_text = response.content.strip()
        
#         # Extract JSON from response
#         # Sometimes LLM adds markdown backticks
#         if "```json" in response_text:
#             response_text = response_text.split("```json")[1].split("```")[0].strip()
#         elif "```" in response_text:
#             response_text = response_text.split("```")[1].split("```")[0].strip()
        
#         # Parse JSON
#         analysis = json.loads(response_text)
        
#         # Validate and set defaults
#         # case_type = analysis.get("case_type", "general")
#         info_provided = analysis.get("info_provided", {})
#         info_needed = analysis.get("info_needed", [])
#         has_sufficient_info = analysis.get("has_sufficient_info", False)
#         user_intent = analysis.get("user_intent", "Seeking legal advice")
        
#         logger.info(f"Query analysis: sufficient_info={has_sufficient_info}, needs={len(info_needed)} items")
        
#         # Safety check for domestic violence
#         # if case_type == "domestic_violence":
#         #     if "current_safety_status" not in info_provided and "safety" not in " ".join(info_needed).lower():
#         #         info_needed.insert(0, "current_safety_status")
#         #         has_sufficient_info = False
        
#         return {
#             # "case_type": case_type,
#             "info_collected": info_provided,
#             "info_needed_list": info_needed,
#             "has_sufficient_info": has_sufficient_info,
#             "user_intent": user_intent
#         }
    
#     except json.JSONDecodeError as e:
#         logger.error(f"JSON parsing error: {str(e)}\nResponse: {response_text[:200]}")
#         # Fallback: basic keyword analysis
#         return fallback_analysis(query)
    
#     except Exception as e:
#         logger.error(f"Query analysis error: {str(e)}")
#         return fallback_analysis(query)

# def fallback_analysis(query: str) -> Dict:
#     """
#     Fallback keyword-based analysis if LLM fails.
#     """
#     query_lower = query.lower()
    
#     # Determine case type
#     if any(word in query_lower for word in ["violence", "abuse", "beat", "assault", "hit", "threat"]):
#         case_type = "domestic_violence"
#     elif any(word in query_lower for word in ["dowry", "dahej", "demand", "harassment"]):
#         case_type = "dowry"
#     elif any(word in query_lower for word in ["custody", "children", "child", "visitation"]):
#         case_type = "child_custody"
#     elif any(word in query_lower for word in ["divorce", "separation", "marriage"]):
#         case_type = "divorce"
#     elif any(word in query_lower for word in ["maintenance", "alimony", "support"]):
#         case_type = "maintenance"
#     else:
#         case_type = "general"
    
#     # Basic information extraction
#     info_provided = {}
#     if "married" in query_lower or "marriage" in query_lower:
#         info_provided["marriage_mentioned"] = "yes"
#     if "child" in query_lower:
#         info_provided["children_mentioned"] = "yes"
    
#     # Determine if we need more info
#     has_sufficient_info = len(query.split()) > 30  # If query is detailed
    
#     info_needed = []
#     if not has_sufficient_info:
#         if case_type == "divorce":
#             info_needed = ["marriage_date", "grounds_for_divorce", "children_details"]
#         elif case_type == "domestic_violence":
#             info_needed = ["current_safety_status", "incident_details", "relationship_to_perpetrator"]
#         elif case_type == "child_custody":
#             info_needed = ["children_ages", "current_custody_arrangement"]
#         else:
#             info_needed = ["detailed_situation", "timeline_of_events"]
    
#     logger.info(f"Fallback analysis: case_type={case_type}")
    
#     return {
#         "case_type": case_type,
#         "info_collected": info_provided,
#         "info_needed_list": info_needed,
#         "has_sufficient_info": has_sufficient_info,
#         "user_intent": "Seeking legal advice"
#     }