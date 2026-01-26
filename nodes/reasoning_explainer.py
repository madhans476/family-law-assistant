"""
Explainable AI - Reasoning Chain Generator

This module generates transparent explanations for legal advice,
showing the step-by-step reasoning process and precedent relevance.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Optional
import os
import json
import logging
from dataclasses import dataclass
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Represents one step in the reasoning chain."""
    step_number: int
    step_type: str  # 'analysis', 'legal_rule', 'precedent', 'conclusion'
    title: str
    explanation: str
    confidence: float
    supporting_sources: List[str]
    legal_provisions: List[str]


@dataclass
class PrecedentRelevance:
    """Explains why a precedent is relevant."""
    precedent_title: str
    similarity_score: float
    matching_factors: List[str]
    different_factors: List[str]
    key_excerpt: str
    relevance_explanation: str
    citation: str


class ReasoningExplainer:
    """
    Generates transparent reasoning chains and precedent explanations.
    """
    
    REASONING_CHAIN_PROMPT = """You are a senior legal analyst. Generate a transparent reasoning chain for the legal advice.

CASE INFORMATION:
User Intent: {user_intent}
Collected Information:
{info_collected}

LEGAL ADVICE PROVIDED:
{response}

RETRIEVED PRECEDENTS:
{precedents_summary}

YOUR TASK:
Generate a step-by-step reasoning chain that explains HOW and WHY you reached the conclusions in the advice.

OUTPUT FORMAT (JSON):
{{
  "reasoning_steps": [
    {{
      "step_number": 1,
      "step_type": "analysis",
      "title": "Situation Analysis",
      "explanation": "Clear explanation of what was identified",
      "confidence": 0.95,
      "supporting_sources": ["source_id_1"],
      "legal_provisions": ["Section X of Act Y"]
    }},
    {{
      "step_number": 2,
      "step_type": "legal_rule",
      "title": "Applicable Law",
      "explanation": "Which laws apply and why",
      "confidence": 1.0,
      "supporting_sources": [],
      "legal_provisions": ["IPC 498A", "DV Act 2005"]
    }},
    {{
      "step_number": 3,
      "step_type": "precedent",
      "title": "Relevant Precedents",
      "explanation": "How similar cases were decided",
      "confidence": 0.85,
      "supporting_sources": ["precedent_1", "precedent_2"],
      "legal_provisions": []
    }},
    {{
      "step_number": 4,
      "step_type": "conclusion",
      "title": "Recommended Action",
      "explanation": "What to do and why",
      "confidence": 0.90,
      "supporting_sources": [],
      "legal_provisions": ["CrPC Section 125"]
    }}
  ],
  "overall_confidence": 0.90,
  "key_assumptions": ["assumption1", "assumption2"],
  "alternative_approaches": ["approach1 if X", "approach2 if Y"]
}}

RULES:
1. Each step must be clear and specific
2. Confidence scores must be realistic (0.0 to 1.0)
3. Link each step to specific sources or laws
4. Include 4-6 reasoning steps
5. Identify any assumptions made
6. Mention alternative approaches if applicable

YOUR REASONING CHAIN (JSON):"""

    PRECEDENT_RELEVANCE_PROMPT = """Analyze why this legal precedent is relevant to the user's case.

USER'S CASE:
{case_summary}

PRECEDENT:
Title: {precedent_title}
Content: {precedent_content}

YOUR TASK:
Explain the similarity and relevance in detail.

OUTPUT FORMAT (JSON):
{{
  "similarity_score": 0.87,
  "matching_factors": [
    "Both involve marriage duration < 1 year",
    "Allegations of physical cruelty",
    "Request for divorce"
  ],
  "different_factors": [
    "Precedent involved dowry, current case doesn't",
    "Precedent had children, current case doesn't"
  ],
  "key_excerpt": "Extract the most relevant 1-2 sentences from precedent",
  "relevance_explanation": "This precedent is relevant because... It supports the advice by...",
  "legal_principle": "The key legal principle is...",
  "outcome_similarity": "The outcome in this precedent suggests..."
}}

RULES:
- Be specific about similarities
- Be honest about differences
- Extract exact quotes for key_excerpt
- Explain practical relevance

YOUR ANALYSIS (JSON):"""
    
    def __init__(self, huggingface_api_key: str = None):
        """Initialize the explainer with LLM."""
        api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                huggingfacehub_api_token=api_key,
                task="text-generation",
                max_new_tokens=2048,
                temperature=0.3,  # Lower for more consistent reasoning
            )
        )
    
    def generate_reasoning_chain(
        self,
        user_intent: str,
        info_collected: Dict[str, str],
        response: str,
        retrieved_chunks: List[Dict]
    ) -> List[ReasoningStep]:
        """
        Generate a transparent reasoning chain for the legal advice.
        
        Returns:
            List of ReasoningStep objects explaining the reasoning
        """
        logger.info("🧠 === GENERATING REASONING CHAIN ===")
        
        try:
            # Format inputs
            info_str = self._format_case_info(info_collected)
            precedents_summary = self._format_precedents_summary(retrieved_chunks)
            
            prompt = self.REASONING_CHAIN_PROMPT.format(
                user_intent=user_intent,
                info_collected=info_str,
                response=response[:1500],  # Truncate to save tokens
                precedents_summary=precedents_summary
            )
            
            conversation = [
                SystemMessage(content="You are a legal reasoning expert. Generate clear, transparent reasoning chains in JSON."),
                HumanMessage(content=prompt)
            ]
            
            llm_response = self.llm.invoke(conversation)
            response_text = llm_response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            
            # Convert to ReasoningStep objects
            reasoning_steps = []
            for step_data in data.get("reasoning_steps", []):
                step = ReasoningStep(
                    step_number=step_data.get("step_number", 0),
                    step_type=step_data.get("step_type", "analysis"),
                    title=step_data.get("title", ""),
                    explanation=step_data.get("explanation", ""),
                    confidence=step_data.get("confidence", 0.8),
                    supporting_sources=step_data.get("supporting_sources", []),
                    legal_provisions=step_data.get("legal_provisions", [])
                )
                reasoning_steps.append(step)
            
            logger.info(f"   ✓ Generated {len(reasoning_steps)} reasoning steps")
            
            # Store additional metadata
            self.overall_confidence = data.get("overall_confidence", 0.85)
            self.key_assumptions = data.get("key_assumptions", [])
            self.alternative_approaches = data.get("alternative_approaches", [])
            
            return reasoning_steps
        
        except Exception as e:
            logger.error(f"❌ Failed to generate reasoning chain: {e}", exc_info=True)
            return self._get_fallback_reasoning()
    
    def explain_precedent_relevance(
        self,
        case_summary: str,
        precedent: Dict
    ) -> PrecedentRelevance:
        """
        Explain why a specific precedent is relevant to the user's case.
        
        Args:
            case_summary: Summary of user's case
            precedent: Dict containing precedent information
            
        Returns:
            PrecedentRelevance object with detailed explanation
        """
        logger.info(f"📚 Analyzing relevance of: {precedent.get('metadata', {}).get('title', 'Unknown')[:50]}...")
        
        try:
            precedent_title = precedent.get("metadata", {}).get("title", "Legal Precedent")
            precedent_content = precedent.get("content", "")[:1000]  # First 1000 chars
            
            prompt = self.PRECEDENT_RELEVANCE_PROMPT.format(
                case_summary=case_summary,
                precedent_title=precedent_title,
                precedent_content=precedent_content
            )
            
            conversation = [
                SystemMessage(content="You are a legal precedent analyst. Explain similarities in JSON format."),
                HumanMessage(content=prompt)
            ]
            
            llm_response = self.llm.invoke(conversation)
            response_text = llm_response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            
            relevance = PrecedentRelevance(
                precedent_title=precedent_title,
                similarity_score=data.get("similarity_score", precedent.get("score", 0.75)),
                matching_factors=data.get("matching_factors", []),
                different_factors=data.get("different_factors", []),
                key_excerpt=data.get("key_excerpt", "")[:300],
                relevance_explanation=data.get("relevance_explanation", ""),
                citation=precedent.get("metadata", {}).get("url", "")
            )
            
            logger.info(f"   ✓ Similarity: {relevance.similarity_score:.0%}")
            
            return relevance
        
        except Exception as e:
            logger.error(f"❌ Failed to explain precedent relevance: {e}", exc_info=True)
            return self._get_fallback_relevance(precedent)
    
    def generate_all_precedent_explanations(
        self,
        case_summary: str,
        retrieved_chunks: List[Dict]
    ) -> List[PrecedentRelevance]:
        """Generate explanations for all retrieved precedents."""
        explanations = []
        
        for precedent in retrieved_chunks[:5]:  # Top 5 precedents
            try:
                explanation = self.explain_precedent_relevance(case_summary, precedent)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain precedent: {e}")
                continue
        
        return explanations
    
    def _format_case_info(self, info_collected: Dict[str, str]) -> str:
        """Format collected information for analysis."""
        if not info_collected:
            return "Limited case information available."
        
        formatted = []
        for key, value in info_collected.items():
            formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)
    
    def _format_precedents_summary(self, retrieved_chunks: List[Dict]) -> str:
        """Format precedents for analysis."""
        if not retrieved_chunks:
            return "No precedents available."
        
        formatted = []
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            formatted.append(f"Precedent {i}: {chunk.get('metadata', {}).get('title', 'Unknown')}")
            formatted.append(f"  Relevance: {chunk.get('score', 0):.0%}")
        
        return "\n".join(formatted)
    
    def _get_fallback_reasoning(self) -> List[ReasoningStep]:
        """Fallback reasoning when generation fails."""
        return [
            ReasoningStep(
                step_number=1,
                step_type="analysis",
                title="Case Analysis",
                explanation="Analyzed your situation based on the information provided.",
                confidence=0.75,
                supporting_sources=[],
                legal_provisions=[]
            ),
            ReasoningStep(
                step_number=2,
                step_type="legal_rule",
                title="Applicable Laws",
                explanation="Identified relevant Indian family laws and provisions.",
                confidence=0.85,
                supporting_sources=[],
                legal_provisions=[]
            ),
            ReasoningStep(
                step_number=3,
                step_type="conclusion",
                title="Recommendation",
                explanation="Generated advice based on legal precedents and your situation.",
                confidence=0.80,
                supporting_sources=[],
                legal_provisions=[]
            )
        ]
    
    def _get_fallback_relevance(self, precedent: Dict) -> PrecedentRelevance:
        """Fallback relevance when analysis fails."""
        return PrecedentRelevance(
            precedent_title=precedent.get("metadata", {}).get("title", "Legal Precedent"),
            similarity_score=precedent.get("score", 0.75),
            matching_factors=["Similar legal context"],
            different_factors=["Specific details may vary"],
            key_excerpt=precedent.get("content", "")[:200],
            relevance_explanation="This precedent was selected based on similarity to your case.",
            citation=precedent.get("metadata", {}).get("url", "")
        )
    
    def format_reasoning_for_response(
        self,
        reasoning_steps: List[ReasoningStep]
    ) -> str:
        """
        Format reasoning chain as human-readable text.
        """
        lines = [
            "\n" + "="*60,
            "🧠 HOW I REACHED THIS CONCLUSION",
            "="*60,
            ""
        ]
        
        for step in reasoning_steps:
            # Step header with confidence
            confidence_emoji = "🟢" if step.confidence >= 0.85 else "🟡" if step.confidence >= 0.70 else "🟠"
            lines.append(f"\n**Step {step.step_number}: {step.title}** {confidence_emoji} ({step.confidence:.0%} confidence)")
            lines.append(step.explanation)
            
            # Legal provisions
            if step.legal_provisions:
                lines.append(f"📜 Legal Basis: {', '.join(step.legal_provisions)}")
            
            # Supporting sources
            if step.supporting_sources:
                lines.append(f"📚 Sources: {len(step.supporting_sources)} precedent(s)")
        
        # Add assumptions if any
        if hasattr(self, 'key_assumptions') and self.key_assumptions:
            lines.extend([
                "",
                "**⚠️ Key Assumptions:**"
            ])
            for assumption in self.key_assumptions:
                lines.append(f"• {assumption}")
        
        # Add alternatives if any
        if hasattr(self, 'alternative_approaches') and self.alternative_approaches:
            lines.extend([
                "",
                "**🔄 Alternative Approaches:**"
            ])
            for alternative in self.alternative_approaches:
                lines.append(f"• {alternative}")
        
        lines.append("="*60)
        
        return "\n".join(lines)
    
    def format_precedent_explanation(
        self,
        precedent_relevance: PrecedentRelevance
    ) -> str:
        """Format precedent explanation as human-readable text."""
        lines = [
            f"\n**{precedent_relevance.precedent_title}**",
            f"Similarity: {precedent_relevance.similarity_score:.0%}",
            "",
            "**Why This Is Relevant:**",
            precedent_relevance.relevance_explanation,
            "",
            "**Matching Factors:**"
        ]
        
        for factor in precedent_relevance.matching_factors:
            lines.append(f"✓ {factor}")
        
        if precedent_relevance.different_factors:
            lines.extend([
                "",
                "**Differences to Note:**"
            ])
            for factor in precedent_relevance.different_factors:
                lines.append(f"⚠ {factor}")
        
        if precedent_relevance.key_excerpt:
            lines.extend([
                "",
                "**Key Excerpt:**",
                f'"{precedent_relevance.key_excerpt}"'
            ])
        
        if precedent_relevance.citation:
            lines.append(f"\n🔗 [View Full Precedent]({precedent_relevance.citation})")
        
        return "\n".join(lines)


def create_case_summary(info_collected: Dict[str, str], user_intent: str) -> str:
    """Create a concise case summary for precedent comparison."""
    summary_parts = [f"Legal Intent: {user_intent}"]
    
    for key, value in info_collected.items():
        summary_parts.append(f"{key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(summary_parts)