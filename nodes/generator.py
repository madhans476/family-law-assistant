"""
Enhanced Generator Node with Explainable AI

Generates legal advice with transparent reasoning chains and precedent explanations.
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
        # repo_id="meta-llama/Llama-3.1-8B-Instruct",
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
    Generate comprehensive legal advice with transparent reasoning.
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
        
        # Check if response seems truncated
        if len(response_content) < 500:
            logger.warning(f"Response seems short: {len(response_content)} chars")
        
        # Initialize explainer
        reasoning_steps = []
        precedent_explanations = []
        
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
                
                # Append reasoning to response
                # reasoning_text = explainer.format_reasoning_for_response(reasoning_steps)
                # response_content += "\n\n" + reasoning_text
                
                # # Append precedent explanations
                # if precedent_explanations:
                #     response_content += "\n\n" + "="*60
                #     response_content += "\n📚 WHY THESE PRECEDENTS MATTER"
                #     response_content += "\n" + "="*60
                    
                #     for i, explanation in enumerate(precedent_explanations, 1):
                #         response_content += f"\n\n**Precedent {i}:**"
                #         response_content += explainer.format_precedent_explanation(explanation)
                
                logger.info(f"   ✓ Added {len(reasoning_steps)} reasoning steps")
                logger.info(f"   ✓ Added {len(precedent_explanations)} precedent explanations")
                
            except Exception as e:
                logger.error(f"Failed to generate reasoning: {e}", exc_info=True)
                # Continue without reasoning rather than failing
        
        # Add outcome prediction if requested
        if include_prediction and info_collected and len(info_collected) >= 30:
            try:
                predictor = CaseOutcomePredictor()
                prediction = predictor.predict_outcome(
                    user_intent=user_intent,
                    info_collected=info_collected,
                    retrieved_precedents=retrieved_chunks
                )
                
                # prediction_text = predictor.format_prediction_for_display(prediction)
                # response_content += "\n\n" + prediction_text
                
                state["prediction"] = {
                    "probability_range": prediction.win_probability_range,
                    "case_strength": prediction.case_strength.value,
                    "confidence": prediction.confidence_level
                }
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
        
        # Add disclaimer
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
        
        return {
            "response": response_content,
            "messages": conversation + [response],
            "reasoning_steps": reasoning_steps_dict,
            "precedent_explanations": precedent_explanations_dict
        }
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {
            "response": f"I apologize, but I encountered an error while generating advice. Please try rephrasing your question or contact support. Error: {str(e)}",
            "messages": messages,
            "reasoning_steps": [],
            "precedent_explanations": []
        }