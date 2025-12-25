from langgraph.graph import StateGraph, START, END
from state import FamilyLawState
from nodes.retriever import retrieve_documents
from nodes.generator import generate_response

def create_graph():
    """Create the family law assistant graph."""
    
    # Initialize graph
    workflow = StateGraph(FamilyLawState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_response)
    
    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app

# Create the compiled graph
family_law_app = create_graph()