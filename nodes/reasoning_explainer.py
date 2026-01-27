"""
Fixed ReasoningExplainer - Returns data structures only, NO text appending.
"""

from typing import List, Dict
import json
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ReasoningStep(BaseModel):
    step_number: int
    step_type: str  # "analysis", "legal_rule", "precedent", "conclusion"
    title: str
    explanation: str
    confidence: float
    supporting_sources: List[str]
    legal_provisions: List[str]


class PrecedentExplanation(BaseModel):
    precedent_title: str
    similarity_score: float
    matching_factors: List[str]
    different_factors: List[str]
    key_excerpt: str
    relevance_explanation: str
    citation: str


class ReasoningExplainer:
    """
    Generates structured explanations of legal reasoning.
    RETURNS DATA STRUCTURES ONLY - DOES NOT MODIFY TEXT.
    """
    
    def generate_reasoning_chain(
        self,
        user_intent: str,
        info_collected: Dict,
        response: str,
        retrieved_chunks: List[Dict]
    ) -> List[ReasoningStep]:
        """
        Generate reasoning steps explaining how the conclusion was reached.
        RETURNS: List of ReasoningStep objects
        DOES NOT: Modify the response text
        """
        reasoning_steps = []
        
        try:
            # Step 1: Situation Analysis
            reasoning_steps.append(ReasoningStep(
                step_number=1,
                step_type="analysis",
                title="Situation Analysis",
                explanation=self._generate_situation_analysis(info_collected, user_intent),
                confidence=0.95,
                supporting_sources=["User Input"],
                legal_provisions=[]
            ))
            
            # Step 2: Legal Rules
            legal_provisions = self._extract_legal_provisions(response)
            reasoning_steps.append(ReasoningStep(
                step_number=2,
                step_type="legal_rule",
                title="Applicable Laws",
                explanation=self._generate_legal_explanation(legal_provisions, user_intent),
                confidence=1.0,
                supporting_sources=["Indian Legal Code"],
                legal_provisions=legal_provisions
            ))
            
            # Step 3: Precedent Analysis
            if retrieved_chunks:
                reasoning_steps.append(ReasoningStep(
                    step_number=3,
                    step_type="precedent",
                    title="Relevant Precedents",
                    explanation=f"Analyzed {len(retrieved_chunks)} relevant precedents to understand similar cases and their outcomes.",
                    confidence=0.80,
                    supporting_sources=[f"Precedent {i+1}" for i in range(min(3, len(retrieved_chunks)))],
                    legal_provisions=[]
                ))
            
            # Step 4: Conclusion
            reasoning_steps.append(ReasoningStep(
                step_number=4,
                step_type="conclusion",
                title="Recommended Course of Action",
                explanation=self._generate_conclusion(user_intent, legal_provisions),
                confidence=0.90,
                supporting_sources=["Legal Analysis"],
                legal_provisions=legal_provisions
            ))
            
            logger.info(f"✓ Generated {len(reasoning_steps)} reasoning steps (structured data only)")
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return []
    
    def generate_all_precedent_explanations(
        self,
        case_summary: str,
        retrieved_chunks: List[Dict]
    ) -> List[PrecedentExplanation]:
        """
        Generate explanations for all retrieved precedents.
        RETURNS: List of PrecedentExplanation objects
        DOES NOT: Modify any text
        """
        explanations = []
        
        try:
            for i, chunk in enumerate(retrieved_chunks[:5]):  # Top 5 precedents
                explanation = self._analyze_precedent(case_summary, chunk, i)
                if explanation:
                    explanations.append(explanation)
            
            logger.info(f"✓ Generated {len(explanations)} precedent explanations (structured data only)")
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating precedent explanations: {e}")
            return []
    
    def _generate_situation_analysis(self, info_collected: Dict, user_intent: str) -> str:
        """Generate situation analysis text."""
        key_facts = []
        
        if "marriage_duration" in info_collected:
            key_facts.append(f"marriage of {info_collected['marriage_duration']}")
        
        if "separation_duration" in info_collected:
            key_facts.append(f"separated for {info_collected['separation_duration']}")
        
        if "emotional_abuse" in str(info_collected.get("abuse", "")).lower():
            key_facts.append("allegations of emotional abuse")
        
        if "child_age" in info_collected:
            key_facts.append(f"minor child aged {info_collected['child_age']}")
        
        if key_facts:
            return f"Identified key case factors: {', '.join(key_facts)}. The case involves {user_intent.lower()}."
        else:
            return f"Analyzed the case involving {user_intent.lower()}."
    
    def _extract_legal_provisions(self, response: str) -> List[str]:
        """Extract legal provisions mentioned in response."""
        provisions = []
        
        # Common patterns
        if "Section 13" in response or "s.13" in response or "s. 13" in response:
            provisions.append("Hindu Marriage Act, 1955 - Section 13")
        
        if "Section 125" in response or "s.125" in response or "s. 125" in response:
            provisions.append("Code of Criminal Procedure - Section 125")
        
        if "498A" in response or "Section 498A" in response:
            provisions.append("Indian Penal Code - Section 498A")
        
        if "Domestic Violence Act" in response or "PWDVA" in response:
            provisions.append("Protection of Women from Domestic Violence Act, 2005")
        
        if "Guardians and Wards Act" in response:
            provisions.append("Guardians and Wards Act, 1890")
        
        return provisions if provisions else ["Indian Family Law"]
    
    def _generate_legal_explanation(self, provisions: List[str], user_intent: str) -> str:
        """Generate explanation of applicable laws."""
        if not provisions:
            return f"Applied general principles of Indian family law relevant to {user_intent.lower()}."
        
        return f"The following laws are applicable: {', '.join(provisions)}. These statutes govern {user_intent.lower()} and related matters."
    
    def _generate_conclusion(self, user_intent: str, provisions: List[str]) -> str:
        """Generate conclusion explanation."""
        return (f"Based on the case analysis and applicable laws ({', '.join(provisions[:2])}), "
                f"recommended specific legal remedies and practical steps for {user_intent.lower()}.")
    
    def _analyze_precedent(
        self,
        case_summary: str,
        chunk: Dict,
        index: int
    ) -> PrecedentExplanation:
        """Analyze a single precedent."""
        try:
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')
            score = chunk.get('score', 0.0)
            
            # Extract key information
            title = metadata.get('title', f'Precedent {index + 1}')
            
            # Simple matching analysis
            matching_factors = self._find_matching_factors(case_summary, content)
            different_factors = self._find_different_factors(case_summary, content)
            
            # Extract key excerpt (first meaningful sentence)
            key_excerpt = self._extract_key_excerpt(content)
            
            # Generate relevance explanation
            relevance = self._generate_relevance_explanation(matching_factors, score)
            
            # Get citation URL
            citation = metadata.get('url', metadata.get('source', ''))
            
            return PrecedentExplanation(
                precedent_title=title,
                similarity_score=score,
                matching_factors=matching_factors,
                different_factors=different_factors,
                key_excerpt=key_excerpt,
                relevance_explanation=relevance,
                citation=citation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing precedent {index}: {e}")
            return None
    
    def _find_matching_factors(self, case_summary: str, content: str) -> List[str]:
        """Find factors that match between case and precedent."""
        factors = []
        
        if "divorce" in case_summary.lower() and "divorce" in content.lower():
            factors.append("Both cases involve divorce proceedings")
        
        if "custody" in case_summary.lower() and "custody" in content.lower():
            factors.append("Both cases address child custody issues")
        
        if "maintenance" in case_summary.lower() and "maintenance" in content.lower():
            factors.append("Both cases involve maintenance claims")
        
        if "abuse" in case_summary.lower() and "abuse" in content.lower():
            factors.append("Both cases involve allegations of abuse")
        
        if not factors:
            factors.append("Similar family law context")
        
        return factors
    
    def _find_different_factors(self, case_summary: str, content: str) -> List[str]:
        """Find factors that differ between case and precedent."""
        factors = []
        
        # Note: This is a simplified version
        if "male" in case_summary.lower() and "wife" in content.lower():
            factors.append("Different party perspectives (husband vs wife)")
        
        if len(factors) < 1:
            factors.append("Different specific circumstances and case details")
        
        return factors
    
    def _extract_key_excerpt(self, content: str) -> str:
        """Extract a key excerpt from the precedent."""
        # Get first meaningful sentence (simplified)
        sentences = content.split('.')
        for sentence in sentences[:3]:
            if len(sentence.strip()) > 50:
                excerpt = sentence.strip()
                if len(excerpt) > 200:
                    excerpt = excerpt[:197] + "..."
                return excerpt
        
        return content[:200] + "..." if len(content) > 200 else content
    
    def _generate_relevance_explanation(self, matching_factors: List[str], score: float) -> str:
        """Generate explanation of why precedent is relevant."""
        if score >= 0.80:
            strength = "highly"
        elif score >= 0.70:
            strength = "significantly"
        else:
            strength = "moderately"
        
        return (f"This precedent is {strength} relevant because: {', '.join(matching_factors[:2])}. "
                f"It provides guidance on similar legal issues and can inform the approach to your case.")


def create_case_summary(info_collected: Dict, user_intent: str) -> str:
    """Create a brief case summary for precedent analysis."""
    parts = [user_intent]
    
    if "user_gender" in info_collected:
        parts.append(f"({info_collected['user_gender']} seeking advice)")
    
    if "marriage_duration" in info_collected:
        parts.append(f"married for {info_collected['marriage_duration']}")
    
    if "child_age" in info_collected:
        parts.append(f"child aged {info_collected['child_age']}")
    
    return " - ".join(parts)