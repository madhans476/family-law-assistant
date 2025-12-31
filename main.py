"""
Complete main.py with proper logging setup at the top.

Key changes:
1. Logging configured FIRST, before other imports
2. Logs to both file and console
3. Creates logs directory automatically
4. Daily log rotation
"""

# ===== LOGGING SETUP - MUST BE FIRST =====
import logging
import sys
from datetime import datetime
from pathlib import Path

# Create logs directory
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

# Create daily log file
log_filename = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# Reduce noise from third-party libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('pymilvus').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("FAMILY LAW ASSISTANT API STARTING")
logger.info(f"Logging to: {log_filename}")
logger.info("="*80)

# ===== NOW IMPORT OTHER MODULES =====
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import json
import os
from graph import family_law_app
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import get_settings
import traceback
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize settings
settings = get_settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Family Law Legal Assistant API",
    description="AI-powered family law consultation system with iterative information gathering",
    version="2.1.0",
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

logger.info(f"CORS origins: {settings.cors_origins}")
logger.info(f"History directory: {settings.history_dir}")

# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, max_length=100)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    sources: List[dict] = []
    conversation_id: str
    message_type: Optional[str] = None
    info_collected: Optional[dict] = None
    info_needed: Optional[List[str]] = None

# Helper Functions
def load_history(conversation_id: str) -> tuple:
    """Load conversation history and state from local file."""
    try:
        history_file = os.path.join(settings.history_dir, f"{conversation_id}.json")
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                messages = []
                for msg in data.get("messages", []):
                    role = msg["role"]
                    content = msg["content"]
                    
                    if role == "HumanMessage":
                        messages.append(HumanMessage(content=content))
                    elif role == "AIMessage":
                        messages.append(AIMessage(content=content))
                    elif role == "SystemMessage":
                        messages.append(SystemMessage(content=content))
                
                state = data.get("state", {})
                logger.info(f"✓ Loaded {len(messages)} messages for {conversation_id}")
                return messages, state
    except Exception as e:
        logger.error(f"❌ Error loading history for {conversation_id}: {str(e)}")
    
    return [], {}


def save_history(conversation_id: str, messages: List, state: dict) -> bool:
    """Save conversation history and state to local file."""
    try:
        history_file = os.path.join(settings.history_dir, f"{conversation_id}.json")
        
        serializable_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                msg_dict = {
                    "role": msg.__class__.__name__,
                    "content": msg.content
                }
                serializable_messages.append(msg_dict)
        
        # Save comprehensive state
        data = {
            "messages": serializable_messages,
            "state": {
                "user_intent": state.get("user_intent"),
                "in_gathering_phase": state.get("in_gathering_phase", False),
                "info_collected": state.get("info_collected", {}),
                "info_needed_list": state.get("info_needed_list", []),
                "gathering_step": state.get("gathering_step", 0),
                "analysis_complete": state.get("analysis_complete", False),
                "has_sufficient_info": state.get("has_sufficient_info", False),
                "current_question_target": state.get("current_question_target"),
                "message_type": state.get("message_type")
            },
            "last_updated": datetime.now().isoformat()
        }
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(serializable_messages)} messages for {conversation_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving history for {conversation_id}: {str(e)}")
        return False

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    logger.info("Root endpoint accessed")
    return {
        "name": "Family Law Legal Assistant API",
        "version": "2.1.0",
        "status": "operational",
        "features": [
            "Iterative information gathering",
            "Intent clarification",
            "Multi-turn conversations",
            "Streaming responses"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0"
    }

@app.post("/chat/stream")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat_stream(request: Request, query_request: QueryRequest):
    """Streaming chat endpoint with proper message saving."""
    
    async def event_generator():
        conversation_id = None
        try:
            conversation_id = query_request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info("="*80)
            logger.info(f"NEW REQUEST: {conversation_id}")
            logger.info(f"Query: {query_request.query}")
            logger.info("="*80)
            
            # Load history and previous state
            messages, previous_state = load_history(conversation_id)
            logger.info(f"Loaded {len(messages)} previous messages")
            
            # Add current user message
            user_message = HumanMessage(content=query_request.query)
            messages.append(user_message)
            logger.info(f"Total messages now: {len(messages)}")
            
            # Prepare state
            state = {
                "query": query_request.query,
                "messages": messages,
                "conversation_id": conversation_id,
                
                # Restore previous state
                "user_intent": previous_state.get("user_intent"),
                "analysis_complete": previous_state.get("analysis_complete", False),
                "needs_clarification": False,
                "clarification_question": None,
                
                # Information gathering state
                "in_gathering_phase": previous_state.get("in_gathering_phase", False),
                "has_sufficient_info": previous_state.get("has_sufficient_info", False),
                "info_collected": previous_state.get("info_collected", {}),
                "info_needed_list": previous_state.get("info_needed_list", []),
                "needs_more_info": False,
                "follow_up_question": None,
                "gathering_step": previous_state.get("gathering_step", 0),
                "current_question_target": previous_state.get("current_question_target"),
                
                # Retrieval and generation
                "retrieved_chunks": [],
                "response": "",
                "sources": [],
                "message_type": None
            }
            
            logger.info(f"State: in_gathering={state['in_gathering_phase']}, "
                       f"step={state['gathering_step']}, "
                       f"collected={len(state['info_collected'])}, "
                       f"needed={len(state['info_needed_list'])}")
            
            yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conversation_id})}\n\n"
            
            # Track response
            accumulated_response = ""
            sources = []
            message_type = None
            final_state = {}
            
            # Stream events from graph
            async for event in family_law_app.astream_events(state, version="v2"):
                kind = event["event"]
                
                # Clarification
                if kind == "on_chain_end" and event.get("name") == "clarify":
                    output = event.get("data", {}).get("output", {})
                    message_type = "clarification"
                    response_text = output.get("response", "")
                    
                    logger.info(f"→ Clarification: {response_text[:100]}...")
                    yield f"data: {json.dumps({'type': 'clarification', 'content': response_text})}\n\n"
                    accumulated_response = response_text
                
                # Information gathering question
                if kind == "on_chain_end" and event.get("name") == "ask_question":
                    output = event.get("data", {}).get("output", {})
                    message_type = "information_gathering"
                    response_text = output.get("response", "")
                    info_collected = output.get("info_collected", {})
                    info_needed = output.get("info_needed", [])
                    
                    logger.info(f"→ Gathering question - collected: {len(info_collected)}, needed: {len(info_needed)}")
                    
                    gathering_data = {
                        'type': 'information_gathering',
                        'content': response_text,
                        'info_collected': info_collected,
                        'info_needed': info_needed
                    }
                    yield f"data: {json.dumps(gathering_data)}\n\n"
                    accumulated_response = response_text
                
                # Retrieval
                if kind == "on_chain_end" and event.get("name") == "retrieve":
                    output = event.get("data", {}).get("output", {})
                    sources = output.get("sources", [])
                    logger.info(f"→ Retrieved {len(sources)} sources")
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                
                # LLM streaming
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        message_type = "final_response"
                        accumulated_response += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                
                # Completion
                if kind == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    final_state = output
            
            # CRITICAL: Save AI message to history
            if accumulated_response:
                ai_message = AIMessage(content=accumulated_response)
                messages.append(ai_message)
                logger.info(f"✓ Added AI message ({len(accumulated_response)} chars), total messages: {len(messages)}")
            
            # Save updated state and messages
            save_history(conversation_id, messages, final_state)
            
            # Send completion
            completion_data = {
                'type': 'done',
                'message_type': message_type or 'final_response',
                'response': accumulated_response
            }
            
            if message_type == "information_gathering":
                completion_data['info_collected'] = final_state.get("info_collected", {})
                completion_data['info_needed'] = final_state.get("info_needed_list", [])
            
            yield f"data: {json.dumps(completion_data)}\n\n"
            
            logger.info(f"✓ Request completed for {conversation_id}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"❌ Error in streaming: {str(e)}")
            logger.error(traceback.format_exc())
            error_data = {'type': 'error', 'message': 'An error occurred.'}
            yield f"data: {json.dumps(error_data)}\n\n"
    
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
    """Get conversation history with state."""
    try:
        logger.info(f"Loading history for: {conversation_id}")
        history_file = os.path.join(settings.history_dir, f"{conversation_id}.json")
        
        if not os.path.exists(history_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"✓ Retrieved history for {conversation_id}")
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
    """Delete conversation history."""
    try:
        logger.info(f"Deleting history for: {conversation_id}")
        history_file = os.path.join(settings.history_dir, f"{conversation_id}.json")
        
        if not os.path.exists(history_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        os.remove(history_file)
        logger.info(f"✓ Deleted {conversation_id}")
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
        logger.info("Listing all conversations")
        conversations = []
        
        if not os.path.exists(settings.history_dir):
            return {"conversations": []}
        
        for filename in os.listdir(settings.history_dir):
            if filename.endswith(".json"):
                conv_id = filename.replace(".json", "")
                filepath = os.path.join(settings.history_dir, filename)
                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        message_count = len(data.get("messages", []))
                        state = data.get("state", {})
                        
                        if state.get("has_sufficient_info"):
                            status = "completed"
                        elif state.get("in_gathering_phase"):
                            status = "gathering_info"
                        else:
                            status = "analyzing"
                        
                        conversations.append({
                            "conversation_id": conv_id,
                            "last_modified": modified_time.isoformat(),
                            "message_count": message_count,
                            "status": status,
                            "user_intent": state.get("user_intent", "Unknown")
                        })
                except:
                    continue
        
        logger.info(f"✓ Found {len(conversations)} conversations")
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
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )