import json
import os
from datetime import datetime
from graph import family_law_app
from langchain_core.messages import HumanMessage

# Local history storage
HISTORY_DIR = "./data/chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def load_history(conversation_id):
    """Load conversation history from local file."""
    history_file = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(conversation_id, messages):
    """Save conversation history to local file."""
    history_file = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    serializable_messages = []
    for msg in messages:
        if hasattr(msg, 'content'):
            serializable_messages.append({
                "role": msg.__class__.__name__,
                "content": msg.content
            })
    
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(serializable_messages, f, indent=2, ensure_ascii=False)

def format_sources(sources):
    """Format sources for display."""
    if not sources:
        return ""
    
    formatted = "\n\n📚 Sources:\n"
    for i, source in enumerate(sources, 1):
        formatted += f"{i}. {source['title']} (Category: {source['category']})\n"
        if source.get('url'):
            formatted += f"   URL: {source['url']}\n"
    
    return formatted

def main():
    """CLI interface for family law assistant."""
    print("=" * 60)
    print("🏛️  Family Law Legal Assistant")
    print("=" * 60)
    print("\nWelcome! I can help you with family law matters including:")
    print("- Divorce cases")
    print("- Husband's cruelty & harassment")
    print("- Domestic violence")
    print("- Dowry disputes")
    print("- Child custody")
    print("- And more...\n")
    
    # Generate conversation ID
    conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"💬 Conversation ID: {conversation_id}")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    messages = []
    
    while True:
        try:
            # Get user input
            query = input("\n👤 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit']:
                print("\n👋 Thank you for using Family Law Assistant. Goodbye!")
                save_history(conversation_id, messages)
                break
            
            # Prepare state
            state = {
                "query": query,
                "messages": messages,
                "conversation_id": conversation_id
            }
            
            print("\n🔍 Searching for relevant information...")
            
            # Run graph
            result = family_law_app.invoke(state)
            
            # Display response
            print("\n🤖 Assistant:", result["response"])
            
            # Display sources
            if result.get("sources"):
                print(format_sources(result["sources"]))
            
            # Update messages for history
            messages = result.get("messages", [])
            
            # Save history after each exchange
            save_history(conversation_id, messages)
            
        except KeyboardInterrupt:
            print("\n\n👋 Conversation interrupted. Goodbye!")
            save_history(conversation_id, messages)
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()