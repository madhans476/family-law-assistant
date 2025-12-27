"""
Production-ready FastAPI server with comprehensive error handling,
logging, and monitoring capabilities.
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import json
import os
from datetime import datetime
from graph import family_law_app
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import get_settings
import logging
import traceback
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Family Law Legal Assistant API",
    description="Production-ready AI-powered family law consultation system",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, max_length=100)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class MessageHistory(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    conversation_id: str
    case_type: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str

# Custom Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Invalid request",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# Helper Functions
def load_history(conversation_id: str) -> List:
    """Load conversation history from local file with error handling."""
    try:
        history_file = os.path.join(settings.history_dir, f"{conversation_id}.json")
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
                logger.info(f"Loaded {len(messages)} messages for conversation {conversation_id}")
                return messages
    except Exception as e:
        logger.error(f"Error loading history for {conversation_id}: {str(e)}")
    return []

def save_history(conversation_id: str, messages: List) -> bool:
    """Save conversation history to local file with error handling."""
    try:
        history_file = os.path.join(settings.history_dir, f"{conversation_id}.json")
        serializable_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                serializable_messages.append({
                    "role": msg.__class__.__name__,
                    "content": msg.content
                })
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(serializable_messages, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(serializable_messages)} messages for conversation {conversation_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving history for {conversation_id}: {str(e)}")
        return False

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Family Law Legal Assistant API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "POST /chat": "Send a query and get a response",
            "POST /chat/stream": "Send a query and get a streaming response",
            "GET /history/{conversation_id}": "Get conversation history",
            "DELETE /history/{conversation_id}": "Delete conversation history",
            "GET /conversations": "List all conversations",
            "GET /health": "Health check endpoint"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat(request: Request, query_request: QueryRequest):
    """
    Non-streaming chat endpoint with comprehensive error handling.
    """
    try:
        conversation_id = query_request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Processing chat request for conversation {conversation_id}")
        
        # Load history
        messages = load_history(conversation_id)
        
        # Prepare state
        state = {
            "query": query_request.query,
            "messages": messages,
            "conversation_id": conversation_id,
            "needs_more_info": True,
            "info_collected": {},
            "case_type": None
        }
        
        # Run graph
        result = family_law_app.invoke(state)
        
        # Save history
        save_history(conversation_id, result.get("messages", []))
        
        logger.info(f"Successfully processed chat request for {conversation_id}")
        
        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            conversation_id=conversation_id,
            case_type=result.get("case_type")
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process your request. Please try again."
        )

@app.post("/chat/stream")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat_stream(request: Request, query_request: QueryRequest):
    """
    Streaming chat endpoint with enhanced error handling.
    """
    async def event_generator():
        try:
            conversation_id = query_request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Processing streaming request for conversation {conversation_id}")
            
            # Load history
            messages = load_history(conversation_id)
            
            # Prepare state with all required fields
            state = {
                "query": query_request.query,
                "messages": messages,
                "conversation_id": conversation_id,
                "case_type": None,
                "user_intent": None,
                "has_sufficient_info": False,
                "info_collected": {},
                "info_needed_list": [],
                "needs_more_info": True,
                "follow_up_question": None,
                "retrieved_chunks": [],
                "response": "",
                "sources": []
            }
            
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conversation_id})}\n\n"
            
            # Stream events
            accumulated_response = ""
            sources = []
            case_type = None
            
            async for event in family_law_app.astream_events(state, version="v2"):
                kind = event["event"]
                
                # Handle retrieval completion
                if kind == "on_chain_end" and event.get("name") == "retrieve":
                    output = event.get("data", {}).get("output", {})
                    sources = output.get("sources", [])
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                
                # Handle information gathering
                if kind == "on_chain_end" and event.get("name") == "gather_info":
                    output = event.get("data", {}).get("output", {})
                    case_type = output.get("case_type")
                    if case_type:
                        yield f"data: {json.dumps({'type': 'case_type', 'case_type': case_type})}\n\n"
                
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
            
            logger.info(f"Successfully completed streaming request for {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred while processing your request.'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str):
    """Get conversation history with validation."""
    try:
        history_file = os.path.join(settings.history_dir, f"{conversation_id}.json")
        
        if not os.path.exists(history_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"Retrieved history for conversation {conversation_id}")
        return data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )

@app.delete("/history/{conversation_id}")
async def delete_history(conversation_id: str):
    """Delete conversation history with validation."""
    try:
        history_file = os.path.join(settings.history_dir, f"{conversation_id}.json")
        
        if not os.path.exists(history_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        os.remove(history_file)
        logger.info(f"Deleted conversation {conversation_id}")
        return {"message": "Conversation history deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation history"
        )

@app.get("/conversations")
async def list_conversations():
    """List all conversations with metadata."""
    try:
        conversations = []
        
        if not os.path.exists(settings.history_dir):
            return {"conversations": []}
        
        for filename in os.listdir(settings.history_dir):
            if filename.endswith(".json"):
                conv_id = filename.replace(".json", "")
                filepath = os.path.join(settings.history_dir, filename)
                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                # Get message count
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        message_count = len(data)
                except:
                    message_count = 0
                
                conversations.append({
                    "conversation_id": conv_id,
                    "last_modified": modified_time.isoformat(),
                    "message_count": message_count
                })
        
        logger.info(f"Retrieved {len(conversations)} conversations")
        return {
            "conversations": sorted(
                conversations,
                key=lambda x: x["last_modified"],
                reverse=True
            )
        }
    
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )