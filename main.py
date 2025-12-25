from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from datetime import datetime
from graph import family_law_app
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

app = FastAPI(title="Family Law Legal Assistant API")

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local history storage
HISTORY_DIR = "./chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# Models
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class MessageHistory(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    conversation_id: str

# Helper functions
def load_history(conversation_id):
    """Load conversation history from local file."""
    history_file = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            messages = []
            for msg in data:
                if msg["role"] == "HumanMessage":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "AIMessage":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "SystemMessage":
                    messages.append(SystemMessage(content=msg["content"]))
            return messages
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

@app.get("/")
async def root():
    return {
        "message": "Family Law Legal Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send a query and get a response",
            "POST /chat/stream": "Send a query and get a streaming response",
            "GET /history/{conversation_id}": "Get conversation history",
            "DELETE /history/{conversation_id}": "Delete conversation history"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """
    Non-streaming chat endpoint.
    """
    try:
        # Generate or use existing conversation ID
        conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load history
        messages = load_history(conversation_id)
        
        # Prepare state
        state = {
            "query": request.query,
            "messages": messages,
            "conversation_id": conversation_id
        }
        
        # Run graph
        result = family_law_app.invoke(state)
        
        # Save history
        save_history(conversation_id, result.get("messages", []))
        
        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            conversation_id=conversation_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    Streaming chat endpoint using astream_events.
    """
    async def event_generator():
        try:
            # Generate or use existing conversation ID
            conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Load history
            messages = load_history(conversation_id)
            
            # Prepare state
            state = {
                "query": request.query,
                "messages": messages,
                "conversation_id": conversation_id
            }
            
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conversation_id})}\n\n"
            
            # Stream events
            accumulated_response = ""
            sources = []
            
            async for event in family_law_app.astream_events(state, version="v2"):
                kind = event["event"]
                
                # Handle retrieval completion
                if kind == "on_chain_end" and event.get("name") == "retrieve":
                    output = event.get("data", {}).get("output", {})
                    sources = output.get("sources", [])
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                
                # Handle LLM streaming tokens
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        accumulated_response += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                
                # Handle completion
                if kind == "on_chain_end" and event.get("name") == "generate":
                    output = event.get("data", {}).get("output", {})
                    messages_to_save = output.get("messages", [])
                    save_history(conversation_id, messages_to_save)
                    yield f"data: {json.dumps({'type': 'done', 'response': accumulated_response})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str):
    """Get conversation history."""
    history_file = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    
    if not os.path.exists(history_file):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    with open(history_file, "r", encoding="utf-8") as f:
        return json.load(f)

@app.delete("/history/{conversation_id}")
async def delete_history(conversation_id: str):
    """Delete conversation history."""
    history_file = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    
    if not os.path.exists(history_file):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    os.remove(history_file)
    return {"message": "Conversation history deleted successfully"}

@app.get("/conversations")
async def list_conversations():
    """List all conversation IDs."""
    conversations = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            conv_id = filename.replace(".json", "")
            filepath = os.path.join(HISTORY_DIR, filename)
            modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            conversations.append({
                "conversation_id": conv_id,
                "last_modified": modified_time.isoformat()
            })
    
    return {"conversations": sorted(conversations, key=lambda x: x["last_modified"], reverse=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)