from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict
from state import FamilyLawState
import os

# Initialize LLM - Llama 3.1 8B Instruct from Hugging Face
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        task="conversational",
    )
)


SYSTEM_PROMPT = """You are a knowledgeable family law legal assistant. Your role is to provide accurate, helpful information about family law matters based on the retrieved case information and legal expertise.

Guidelines:
- Provide clear, practical advice based on the context provided
- Reference relevant cases or legal principles when applicable
- Be empathetic and professional, as family law matters are often sensitive
- If the retrieved information doesn't fully address the query, acknowledge this and provide general guidance
- Always clarify that this is informational and not a substitute for personalized legal advice
- Format your response clearly with proper structure

Never make up information. Only use the provided context."""

def format_context(retrieved_chunks):
    """Format retrieved chunks into context string."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Source {i}] (Category: {chunk['metadata']['category']})")
        context_parts.append(f"Title: {chunk['metadata']['title']}")
        context_parts.append(f"Content: {chunk['content']}")
        context_parts.append(f"Relevance Score: {chunk['score']:.3f}\n")
    
    return "\n".join(context_parts)

def generate_response(state: FamilyLawState) -> Dict:
    """
    Generate a response using Claude based on retrieved context.
    """
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    messages = state.get("messages", [])
    
    if not retrieved_chunks:
        return {
            "response": "I apologize, but I couldn't find relevant information in the family law database to answer your question. Please try rephrasing your question or ask about topics like divorce, custody, domestic violence, dowry, or marital cruelty."
        }
    
    # Format context
    context = format_context(retrieved_chunks)
    
    # Build conversation with history
    conversation = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Add previous messages (history)
    if messages:
        conversation.extend(messages)
    
    # Add current query with context
#     user_message = f"""Based on the following retrieved information from family law cases, please answer the user's question:

#       RETRIEVED CONTEXT:
#       {context}

#       USER QUESTION: {query}

#       Provide a comprehensive, helpful response based on the above context."""
    
    example_query = ("I got engaged in June 2018 and married in February 2019. Soon after, my husband stopped caring for me and the household, then left. "
                         "He abused me for talking to friends and about my past, which he already knew. Yesterday, he called me to meet, and I hoped we could reconcile. "
                         "Instead, he beat me and stopped me from leaving. I escaped this morning. I want a divorce as soon as possible.")
    
    example_responses = [
            "First, lodge an FIR under Section 498A for cruelty. Then consult a lawyer to file two cases: one under the Domestic Violence Act, and another for Judicial Separation under Section 10 of the Hindu Marriage Act, since you can’t seek divorce before one year of marriage.",
            "You can file for divorce in family court under the Hindu Marriage Act on grounds of mental and physical cruelty. Also, register an FIR under Sections 498A and 323 IPC.",
            "Since you married in February 2019, you must wait one year before filing for divorce. Meanwhile, file a police complaint for assault — it will support your case for cruelty. You can also claim maintenance under Section 125 CrPC if he isn’t supporting you."
        ]
    
    prompt = (
            "You are a senior Indian family law analyst. Follow the guidelines strictly. "
            "Explain legal issues in simple, plain English that anyone can understand. Use a warm, professional tone, and avoid robotic phrasing. "
            "Do not include reasoning scaffolds, JSON, or any pre/post text.\n\n"
            "Guidelines:\n"
            "- Internally reason as Issue → Rule (statute/precedent) → Application → Conclusion, but only output the detailed final answer.\n"
            "- Prefer authoritative Indian sources and cite succinctly, e.g., (IPC s.498A), (HMA 1955 s.13), (CrPC s.125).\n"
            "- If a precise section is uncertain, mention it briefly without guessing.\n"
            "- Give a concise response to ensure a Flesch Reading Ease score of atleast 55.\n"
            "- Explain legal terms briefly in everyday language when necessary.\n"
            "- Give practical guidance wherever possible, focusing on what a person can realistically do.\n\n"
            "Style Example (tone only, not ground truth):\n"
            f"Query:\n{example_query}\n\n"
            "Example Responses:\n"
            f"1. {example_responses[0]}\n"
            f"2. {example_responses[1]}\n"
            f"3. {example_responses[2]}\n"

            f"Context:\n{context}\n\n"
            "Now answer this user query with the detailed final answer only:\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )
    
    conversation.append(HumanMessage(content=prompt))
    
    # Generate response
    response = llm.invoke(conversation)
    
    return {
        "response": response.content,
        "messages": conversation + [response]
    }