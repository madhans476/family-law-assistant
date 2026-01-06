"""
Production-grade case outcome prediction system.

This system analyzes case details and provides win probability predictions
based on legal precedents, case strength factors, and historical data.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Optional, Tuple
import os
import logging
from dataclasses import dataclass
from enum import Enum
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logger = logging.getLogger(__name__)


class CaseStrength(Enum):
    """Case strength categories."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"


@dataclass
class OutcomePrediction:
    """Structured outcome prediction result."""
    win_probability_range: Tuple[int, int]  # e.g., (60, 75) means 60-75%
    confidence_level: str  # high, medium, low
    case_strength: CaseStrength
    favorable_factors: List[str]
    unfavorable_factors: List[str]
    key_precedents: List[str]
    recommendations: List[str]
    disclaimers: List[str]


class CaseOutcomePredictor:
    """
    Predicts case outcomes based on collected information and legal precedents.
    
    IMPORTANT: This is for informational purposes only and should never be
    presented as a guarantee or professional legal prediction.
    """
    
    PREDICTION_PROMPT = """You are a senior Indian family law expert analyzing case outcomes based on precedents and case factors.

TASK: Analyze the case and provide a structured assessment of likely outcomes.

CASE INFORMATION:
Intent: {user_intent}
Collected Information:
{info_collected}

RELEVANT PRECEDENTS:
{precedents}

YOUR ANALYSIS MUST BE STRUCTURED AS FOLLOWS:

1. WIN PROBABILITY ESTIMATE:
   - Provide a range (e.g., 40-60%, 65-80%)
   - Consider: evidence strength, legal grounds, precedents, procedural factors
   - Be conservative and realistic

2. CASE STRENGTH FACTORS:
   FAVORABLE FACTORS (what helps the case):
   - List specific strengths
   
   UNFAVORABLE FACTORS (what weakens the case):
   - List specific weaknesses

3. KEY PRECEDENTS:
   - Cite 2-3 most relevant precedents
   - Explain how they support or challenge the case

4. RECOMMENDATIONS:
   - What would strengthen the case
   - What evidence to gather
   - Procedural considerations

CRITICAL RULES:
- Never guarantee outcomes
- Be honest about weaknesses
- Consider jurisdiction-specific factors
- Account for judge discretion
- Mention settlement possibilities where relevant

OUTPUT FORMAT (JSON):
{{
  "probability_range_min": <number 0-100>,
  "probability_range_max": <number 0-100>,
  "confidence": "high|medium|low",
  "case_strength": "very_strong|strong|moderate|weak|very_weak",
  "favorable_factors": ["factor1", "factor2", ...],
  "unfavorable_factors": ["factor1", "factor2", ...],
  "key_precedents": ["precedent1", "precedent2", ...],
  "recommendations": ["rec1", "rec2", ...],
  "analysis_notes": "brief summary of key considerations"
}}

YOUR ANALYSIS:"""
    
    def __init__(self, huggingface_api_key: str = None):
        """Initialize the predictor with LLM."""
        api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=api_key,
                task="text-generation",
                max_new_tokens=2048,
                temperature=0.5,  # Lower temperature for more conservative predictions
            )
        )
    
    def predict_outcome(
        self,
        user_intent: str,
        info_collected: Dict[str, str],
        retrieved_precedents: List[Dict],
        include_in_response: bool = True
    ) -> OutcomePrediction:
        """
        Predict case outcome with win probability.
        
        Args:
            user_intent: The legal intent/goal
            info_collected: Dictionary of collected case information
            retrieved_precedents: List of relevant legal precedents
            include_in_response: Whether to include this in main response
        
        Returns:
            OutcomePrediction object with structured prediction
        """
        logger.info("🎯 === PREDICTING CASE OUTCOME ===")
        
        try:
            # Format inputs
            info_str = self._format_case_info(info_collected)
            precedents_str = self._format_precedents(retrieved_precedents)
            
            # Build prompt
            prompt = self.PREDICTION_PROMPT.format(
                user_intent=user_intent,
                info_collected=info_str,
                precedents=precedents_str
            )
            
            # Get prediction
            conversation = [
                SystemMessage(content="You are an expert legal analyst. Provide honest, realistic case assessments in JSON format."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(conversation)
            response_text = response.content.strip()
            
            # Parse JSON response
            import json
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            prediction_data = json.loads(response_text)
            
            # Build structured prediction
            prediction = OutcomePrediction(
                win_probability_range=(
                    prediction_data.get("probability_range_min", 40),
                    prediction_data.get("probability_range_max", 60)
                ),
                confidence_level=prediction_data.get("confidence", "medium"),
                case_strength=CaseStrength(prediction_data.get("case_strength", "moderate")),
                favorable_factors=prediction_data.get("favorable_factors", []),
                unfavorable_factors=prediction_data.get("unfavorable_factors", []),
                key_precedents=prediction_data.get("key_precedents", []),
                recommendations=prediction_data.get("recommendations", []),
                disclaimers=self._get_disclaimers()
            )
            
            logger.info(f"   Prediction: {prediction.win_probability_range[0]}-{prediction.win_probability_range[1]}%")
            logger.info(f"   Strength: {prediction.case_strength.value}")
            logger.info(f"   Confidence: {prediction.confidence_level}")
            
            return prediction
        
        except Exception as e:
            logger.error(f"❌ Outcome prediction failed: {e}", exc_info=True)
            return self._get_fallback_prediction()
    
    def _format_case_info(self, info_collected: Dict[str, str]) -> str:
        """Format collected information for analysis."""
        if not info_collected:
            return "Limited case information available."
        
        formatted = []
        for key, value in info_collected.items():
            formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)
    
    def _format_precedents(self, retrieved_precedents: List[Dict]) -> str:
        """Format precedents for analysis."""
        if not retrieved_precedents:
            return "No specific precedents available for analysis."
        
        formatted = []
        for i, precedent in enumerate(retrieved_precedents[:5], 1):
            formatted.append(f"\n[Precedent {i}]")
            formatted.append(f"Title: {precedent.get('metadata', {}).get('title', 'Unknown')}")
            formatted.append(f"Relevance: {precedent.get('score', 0):.0%}")
            formatted.append(f"Summary: {precedent.get('content', '')[:300]}...")
        
        return "\n".join(formatted)
    
    def _get_disclaimers(self) -> List[str]:
        """Get standard disclaimers for predictions."""
        return [
            "This prediction is based on analysis of precedents and case factors, not a guarantee of outcome.",
            "Actual case outcomes depend on many factors including: quality of legal representation, evidence presentation, judge discretion, and procedural factors.",
            "Legal predictions are inherently uncertain. Always consult with a qualified attorney.",
            "Settlement or alternative dispute resolution may be preferable to litigation in many cases.",
            "Court procedures and outcomes can vary significantly by jurisdiction and specific court."
        ]
    
    def _get_fallback_prediction(self) -> OutcomePrediction:
        """Fallback prediction when analysis fails."""
        return OutcomePrediction(
            win_probability_range=(30, 70),
            confidence_level="low",
            case_strength=CaseStrength.MODERATE,
            favorable_factors=["Unable to analyze - insufficient data"],
            unfavorable_factors=["Prediction system error - manual review needed"],
            key_precedents=[],
            recommendations=[
                "Consult with a family law attorney for proper case assessment",
                "Gather all relevant evidence and documentation",
                "Consider mediation or settlement options"
            ],
            disclaimers=self._get_disclaimers()
        )
    
    def format_prediction_for_display(self, prediction: OutcomePrediction) -> str:
        """
        Format prediction as human-readable text for inclusion in response.
        """
        lines = [
            "\n" + "="*60,
            "📊 CASE OUTCOME ASSESSMENT",
            "="*60,
            "",
            f"**Estimated Win Probability:** {prediction.win_probability_range[0]}-{prediction.win_probability_range[1]}%",
            f"**Case Strength:** {prediction.case_strength.value.replace('_', ' ').title()}",
            f"**Confidence Level:** {prediction.confidence_level.title()}",
            "",
            "**Favorable Factors (What Helps Your Case):**"
        ]
        
        for factor in prediction.favorable_factors:
            lines.append(f"✓ {factor}")
        
        lines.extend([
            "",
            "**Unfavorable Factors (Challenges to Address):**"
        ])
        
        for factor in prediction.unfavorable_factors:
            lines.append(f"✗ {factor}")
        
        if prediction.key_precedents:
            lines.extend([
                "",
                "**Relevant Precedents:**"
            ])
            for precedent in prediction.key_precedents:
                lines.append(f"• {precedent}")
        
        lines.extend([
            "",
            "**Recommendations to Strengthen Your Case:**"
        ])
        
        for rec in prediction.recommendations:
            lines.append(f"→ {rec}")
        
        lines.extend([
            "",
            "⚠️ **IMPORTANT DISCLAIMERS:**"
        ])
        
        for disclaimer in prediction.disclaimers:
            lines.append(f"• {disclaimer}")
        
        lines.append("="*60)
        
        return "\n".join(lines)


# Integration with generator node
def generate_response_with_prediction(state) -> Dict:
    """
    Enhanced generator that includes outcome prediction.
    
    Add this to your generator.py:
    """
    from nodes.generator import generate_response as original_generate
    
    # Generate main response
    result = original_generate(state)
    
    # Check if we should include prediction
    retrieved_chunks = state.get("retrieved_chunks", [])
    info_collected = state.get("info_collected", {})
    user_intent = state.get("user_intent", "")
    
    # Only predict if we have sufficient information
    if retrieved_chunks and info_collected and len(info_collected) >= 3:
        try:
            predictor = CaseOutcomePredictor()
            prediction = predictor.predict_outcome(
                user_intent=user_intent,
                info_collected=info_collected,
                retrieved_precedents=retrieved_chunks
            )
            
            # Format and append to response
            prediction_text = predictor.format_prediction_for_display(prediction)
            result["response"] += "\n\n" + prediction_text
            
            # Add prediction to metadata
            result["prediction"] = {
                "probability_range": prediction.win_probability_range,
                "case_strength": prediction.case_strength.value,
                "confidence": prediction.confidence_level
            }
            
        except Exception as e:
            logger.error(f"Failed to generate prediction: {e}")
            # Continue without prediction rather than failing
    
    return result