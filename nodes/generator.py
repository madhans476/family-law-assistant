"""
Fixed Generator Node - Does NOT append reasoning to response text.

Key fixes:
1. Reasoning and citations returned separately in state
2. NOT appended to response_content
3. Clean response without scaffold text
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict
from state import FamilyLawState
import os
import logging
from nodes.case_outcome_predictor import CaseOutcomePredictor
from nodes.reasoning_explainer import (
    ReasoningExplainer, 
    create_case_summary
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        task="conversational",
        max_new_tokens=2048,
        temperature=0.7,
    )
)

example_query = ("I got engaged in June 2018 and married in February 2019. Soon after, my husband stopped caring for me and the household, then left. "
"He abused me for talking to friends and about my past, which he already knew. Yesterday, he called me to meet, and I hoped we could reconcile. "
"Instead, he beat me and stopped me from leaving. I escaped this morning. I want a divorce as soon as possible.")

example_responses = [
"First, lodge an FIR under Section 498A for cruelty. Then consult a lawyer to file two cases: one under the Domestic Violence Act, and another for Judicial Separation under Section 10 of the Hindu Marriage Act, since you can't seek divorce before one year of marriage.",
"You can file for divorce in family court under the Hindu Marriage Act on grounds of mental and physical cruelty. Also, register an FIR under Sections 498A and 323 IPC.",
"Since you married in February 2019, you must wait one year before filing for divorce. Meanwhile, file a police complaint for assault — it will support your case for cruelty. You can also claim maintenance under Section 125 CrPC if he isn't supporting you."
]

SYSTEM_PROMPT = (
"You are a senior Indian law analyst. Follow the guidelines strictly. "
"Explain legal issues in simple, plain English that anyone can understand. Use a warm, professional tone, and avoid robotic phrasing. "
"Do not include reasoning scaffolds, JSON, or any pre/post text.\n\n"
"Guidelines:\n"
"- Internally reason as Issue → Rule (statute/precedent) → Application → Conclusion, but only output the detailed final answer.\n"
"- Prefer authoritative Indian sources and cite succinctly, e.g., (IPC s.498A), (HMA 1955 s.13), (CrPC s.125).\n"
"- If a precise section is uncertain, mention it briefly without guessing.\n"
"- Give concise response to ensure a Flesch Reading Ease score of atleast 55+.\n"
"- Explain legal terms briefly in everyday language when necessary.\n"
"- Give practical guidance wherever possible, focusing on what a person can realistically do.\n\n"
"Style Example (tone only, not ground truth):\n"
f"Query:\n{example_query}\n\n"
"Example Responses:\n"
f"1. {example_responses[0]}\n"
f"2. {example_responses[1]}\n"
f"3. {example_responses[2]}\n"
)

def format_context(retrieved_chunks: list) -> str:
    """Format retrieved chunks efficiently."""
    if not retrieved_chunks:
        return "No relevant precedents found."
    
    context_parts = ["RELEVANT LEGAL PRECEDENTS:\n"]

    for i, chunk in enumerate(retrieved_chunks[:5], 1):
        context_parts.append(f"\n[Precedent {i}] ({chunk['score']:.0%} relevance)")
        context_parts.append(f"Title: {chunk['metadata']['title']}")
        context_parts.append(f"Category: {chunk['metadata']['category']}")
        content = chunk['content']
        context_parts.append(f"Content: {content}")
        context_parts.append("")
    
    return "\n".join(context_parts)

def format_case_info(info_collected: Dict, user_intent: str) -> str:
    """Format collected case information."""
    if not info_collected:
        return "Limited case information available."
    
    case_summary = [f"CASE: {user_intent.upper()}\n"]
    case_summary.append("CLIENT INFORMATION:")
    
    for key, value in info_collected.items():
        case_summary.append(f"• {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(case_summary)

def generate_response(state: FamilyLawState) -> Dict:
    """
    Generate legal advice WITHOUT appending reasoning to response text.
    Reasoning returned separately in state.
    """
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    messages = state.get("messages", [])
    info_collected = state.get("info_collected", {})
    user_intent = state.get("user_intent", "legal advice")
    include_prediction = state.get("include_prediction", True)
    include_reasoning = state.get("include_reasoning", True)
    
    # Validate we have information
    if not retrieved_chunks:
        logger.warning("No retrieved chunks available for generation")
        return {
            "response": "I apologize, but I couldn't find sufficient relevant information in the legal database to provide comprehensive advice for your specific situation. Please consider consulting with a family law attorney directly for personalized guidance.",
            "messages": messages,
            "reasoning_steps": [],
            "precedent_explanations": []
        }
    
    # Format context and case information
    legal_context = format_context(retrieved_chunks)
    case_information = format_case_info(info_collected, user_intent)
    
    # Build conversation
    conversation = [SystemMessage(content=SYSTEM_PROMPT)]
    
    if messages:
        conversation.extend(messages[-4:])
    
    # Construct prompt
    prompt = f"""Provide complete legal advice (experienced lawyer) based on the case information and precedents below.

{case_information}

{legal_context}

CLIENT QUERY: {query}

Provide a COMPLETE, well-structured response with:
- Internally reason as Issue → Rule (statute/precedent) → Application → Conclusion, but only output the detailed final answer.
- Prefer authoritative Indian sources and cite succinctly, e.g., (IPC s.498A), (HMA 1955 s.13), (CrPC s.125).
- If a precise section is uncertain, mention it briefly without guessing.
- Give concise response to ensure a Flesch Reading Ease score of atleast 55+.
- Explain legal terms briefly in everyday language when necessary.
- Give practical guidance wherever possible, focusing on what a person can realistically do.

Use empathetic, professional, simple language. Be thorough - this is important for the client's case.

YOUR COMPLETE RESPONSE:"""
    
    conversation.append(HumanMessage(content=prompt))
    
    try:
        logger.info("Generating response with explainable AI")
        
        # Generate main response
        response = llm.invoke(conversation)
        response_content = response.content
        
        # *** CRITICAL CLEANING: Remove any appended reasoning/JSON ***
        # Check for common patterns that indicate appended content
        cleanup_markers = [
            "Here is the generated reasoning",
            "Here is the analysis in JSON",
            "Here is my analysis in JSON",
            "```json",
            '{"reasoning_steps"',
            '{"similarity_score"',
            "WIN PROBABILITY ESTIMATE:",
            "CASE STRENGTH FACTORS:"
        ]
        
        for marker in cleanup_markers:
            if marker in response_content:
                response_content = response_content.split(marker)[0].strip()
                logger.warning(f"⚠️ Removed appended content after marker: {marker}")
        
        # Remove any trailing JSON
        if response_content.rstrip().endswith('}'):
            # Check if last 500 chars look like JSON
            tail = response_content[-500:]
            if tail.count('{') > 2 or tail.count('"') > 10:
                # Find the last sentence before JSON starts
                sentences = response_content.split('.')
                clean_sentences = []
                for sentence in sentences:
                    if '{' not in sentence and '"reasoning' not in sentence.lower():
                        clean_sentences.append(sentence)
                    else:
                        break
                response_content = '.'.join(clean_sentences) + '.'
                logger.warning("⚠️ Removed trailing JSON from response")
        
        # Check if response seems truncated
        if len(response_content) < 500:
            logger.warning(f"Response seems short: {len(response_content)} chars")
        
        # Initialize explainer
        reasoning_steps = []
        precedent_explanations = []
        
        # Generate reasoning separately (NOT appended to response)
        if include_reasoning:
            try:
                logger.info("🧠 Generating reasoning chain...")
                explainer = ReasoningExplainer()
                
                # Generate reasoning chain
                reasoning_steps = explainer.generate_reasoning_chain(
                    user_intent=user_intent,
                    info_collected=info_collected,
                    response=response_content,
                    retrieved_chunks=retrieved_chunks
                )
                
                # Generate precedent explanations
                case_summary = create_case_summary(info_collected, user_intent)
                precedent_explanations = explainer.generate_all_precedent_explanations(
                    case_summary=case_summary,
                    retrieved_chunks=retrieved_chunks
                )
                
                # *** CRITICAL: DO NOT APPEND TO response_content ***
                # The reasoning is returned separately in the state
                # DO NOT modify response_content here
                
                logger.info(f"   ✓ Generated {len(reasoning_steps)} reasoning steps (NOT appended)")
                logger.info(f"   ✓ Generated {len(precedent_explanations)} precedent explanations (NOT appended)")
                
            except Exception as e:
                logger.error(f"Failed to generate reasoning: {e}", exc_info=True)
        
        # *** CRITICAL CHECK: Ensure nothing was accidentally appended ***
        # Remove any JSON or reasoning text that might have been added
        if "reasoning_steps" in response_content or "{" in response_content[-500:]:
            # Find where the actual legal advice ends (before any JSON)
            if "Here is the generated reasoning" in response_content:
                response_content = response_content.split("Here is the generated reasoning")[0].strip()
            elif "```json" in response_content:
                response_content = response_content.split("```json")[0].strip()
            elif '{"reasoning_steps"' in response_content:
                response_content = response_content.split('{"reasoning_steps"')[0].strip()
            
            logger.warning("⚠️ Removed accidentally appended reasoning from response_content")
        
        # Add outcome prediction if requested (also NOT appended)
        prediction_data = None
        # if include_prediction and info_collected and len(info_collected) >= 30:
        #     try:
        #         predictor = CaseOutcomePredictor()
        #         prediction = predictor.predict_outcome(
        #             user_intent=user_intent,
        #             info_collected=info_collected,
        #             retrieved_precedents=retrieved_chunks
        #         )
                
        #         # Store prediction separately, NOT in response text
        #         prediction_data = {
        #             "probability_range": prediction.win_probability_range,
        #             "case_strength": prediction.case_strength.value,
        #             "confidence": prediction.confidence_level
        #         }
                
        #         logger.info(f"   ✓ Generated prediction: {prediction.case_strength.value}")
        #     except Exception as e:
        #         logger.error(f"Prediction failed: {e}")
        
        # Add disclaimer to response
        if "not a substitute for legal advice" not in response_content.lower():
            response_content += "\n\n---\n**Disclaimer**: This information is for educational purposes only and does not constitute legal advice. Please consult with a qualified family law attorney for personalized legal guidance."
        
        logger.info(f"Generated response: {len(response_content)} characters")
        
        # Convert reasoning steps to serializable format
        reasoning_steps_dict = []
        for step in reasoning_steps:
            reasoning_steps_dict.append({
                "step_number": step.step_number,
                "step_type": step.step_type,
                "title": step.title,
                "explanation": step.explanation,
                "confidence": step.confidence,
                "supporting_sources": step.supporting_sources,
                "legal_provisions": step.legal_provisions
            })
        
        # Convert precedent explanations to serializable format
        precedent_explanations_dict = []
        for explanation in precedent_explanations:
            precedent_explanations_dict.append({
                "precedent_title": explanation.precedent_title,
                "similarity_score": explanation.similarity_score,
                "matching_factors": explanation.matching_factors,
                "different_factors": explanation.different_factors,
                "key_excerpt": explanation.key_excerpt,
                "relevance_explanation": explanation.relevance_explanation,
                "citation": explanation.citation
            })
        
        # Return clean response with reasoning/citations separate
        return {
            "response": response_content,  # CLEAN - No reasoning appended
            "messages": conversation + [response],
            "reasoning_steps": reasoning_steps_dict,  # Separate
            "precedent_explanations": precedent_explanations_dict,  # Separate
            "prediction": prediction_data  # Separate
        }
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {
            "response": f"I apologize, but I encountered an error while generating advice. Please try rephrasing your question or contact support. Error: {str(e)}",
            "messages": messages,
            "reasoning_steps": [],
            "precedent_explanations": []
        }