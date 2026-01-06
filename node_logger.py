"""
Production-grade node execution logger for legal assistant.

This module creates detailed execution logs for every node in the graph,
enabling comprehensive evaluation and debugging.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
import traceback

logger = logging.getLogger(__name__)


class NodeExecutionLogger:
    """
    Logs detailed execution information for each node in the graph.
    Creates structured logs for analysis and evaluation.
    """
    
    def __init__(self, base_log_dir: str = "./logs/executions"):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_conversation_log_dir(self, conversation_id: str) -> Path:
        """Get or create directory for conversation logs."""
        conv_dir = self.base_log_dir / conversation_id
        conv_dir.mkdir(exist_ok=True)
        return conv_dir
    
    def log_node_execution(
        self,
        conversation_id: str,
        node_name: str,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        execution_time: float,
        error: Optional[Exception] = None
    ) -> None:
        """
        Log a single node execution with full context.
        
        Args:
            conversation_id: Unique conversation identifier
            node_name: Name of the executed node
            input_state: State before node execution
            output_state: State after node execution
            execution_time: Execution duration in seconds
            error: Any exception that occurred
        """
        try:
            conv_dir = self.get_conversation_log_dir(conversation_id)
            timestamp = datetime.now().isoformat()
            
            # Create execution log entry
            log_entry = {
                "timestamp": timestamp,
                "node_name": node_name,
                "execution_time_seconds": round(execution_time, 3),
                "success": error is None,
                "input_state": self._serialize_state(input_state),
                "output_state": self._serialize_state(output_state),
                "state_changes": self._compute_state_changes(input_state, output_state),
            }
            
            if error:
                log_entry["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc()
                }
            
            # Save to node-specific log file
            node_log_file = conv_dir / f"{node_name}_executions.jsonl"
            with open(node_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            # Update conversation summary
            self._update_conversation_summary(conversation_id, node_name, log_entry)
            
            logger.info(f"Logged execution: {conversation_id}/{node_name} ({execution_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Failed to log node execution: {e}", exc_info=True)
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for JSON storage, handling special types."""
        serialized = {}
        
        for key, value in state.items():
            try:
                if key == "messages":
                    # Serialize message objects
                    serialized[key] = [
                        {
                            "role": msg.__class__.__name__,
                            "content": msg.content[:500] if hasattr(msg, 'content') else str(msg)[:500]
                        }
                        for msg in (value or [])
                    ]
                elif isinstance(value, (str, int, float, bool, type(None))):
                    serialized[key] = value
                elif isinstance(value, (list, dict)):
                    serialized[key] = value
                else:
                    serialized[key] = str(value)[:500]
            except Exception as e:
                serialized[key] = f"<serialization_error: {e}>"
        
        return serialized
    
    def _compute_state_changes(
        self, 
        input_state: Dict[str, Any], 
        output_state: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute what changed between input and output states."""
        changes = {}
        
        all_keys = set(input_state.keys()) | set(output_state.keys())
        
        for key in all_keys:
            input_val = input_state.get(key)
            output_val = output_state.get(key)
            
            # Skip messages for change detection (too verbose)
            if key == "messages":
                changes[key] = {
                    "count_before": len(input_val) if input_val else 0,
                    "count_after": len(output_val) if output_val else 0
                }
                continue
            
            if input_val != output_val:
                changes[key] = {
                    "before": self._safe_serialize(input_val),
                    "after": self._safe_serialize(output_val)
                }
        
        return changes
    
    def _safe_serialize(self, value: Any) -> Any:
        """Safely serialize a value for comparison."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, dict)):
            return value
        else:
            return str(value)[:200]
    
    def _update_conversation_summary(
        self, 
        conversation_id: str, 
        node_name: str, 
        log_entry: Dict[str, Any]
    ) -> None:
        """Update high-level conversation summary."""
        conv_dir = self.get_conversation_log_dir(conversation_id)
        summary_file = conv_dir / "execution_summary.json"
        
        # Load existing summary
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            summary = {
                "conversation_id": conversation_id,
                "started_at": datetime.now().isoformat(),
                "node_executions": [],
                "total_execution_time": 0,
                "node_counts": {}
            }
        
        # Update summary
        summary["node_executions"].append({
            "node_name": node_name,
            "timestamp": log_entry["timestamp"],
            "execution_time": log_entry["execution_time_seconds"],
            "success": log_entry["success"]
        })
        summary["total_execution_time"] += log_entry["execution_time_seconds"]
        summary["node_counts"][node_name] = summary["node_counts"].get(node_name, 0) + 1
        summary["last_updated"] = datetime.now().isoformat()
        
        # Save summary
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def create_evaluation_report(self, conversation_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for a conversation.
        
        Returns:
            Dict containing analysis of all node executions
        """
        conv_dir = self.get_conversation_log_dir(conversation_id)
        
        if not conv_dir.exists():
            return {"error": "Conversation not found"}
        
        # Load summary
        summary_file = conv_dir / "execution_summary.json"
        if not summary_file.exists():
            return {"error": "No execution summary found"}
        
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        # Analyze node executions
        report = {
            "conversation_id": conversation_id,
            "summary": summary,
            "node_analysis": {},
            "timeline": []
        }
        
        # Load and analyze each node's executions
        for node_file in conv_dir.glob("*_executions.jsonl"):
            node_name = node_file.stem.replace("_executions", "")
            executions = []
            
            with open(node_file, "r", encoding="utf-8") as f:
                for line in f:
                    executions.append(json.loads(line))
            
            report["node_analysis"][node_name] = {
                "total_executions": len(executions),
                "total_time": sum(e["execution_time_seconds"] for e in executions),
                "success_rate": sum(1 for e in executions if e["success"]) / len(executions),
                "executions": executions
            }
            
            # Add to timeline
            for exec_log in executions:
                report["timeline"].append({
                    "timestamp": exec_log["timestamp"],
                    "node": node_name,
                    "duration": exec_log["execution_time_seconds"],
                    "success": exec_log["success"]
                })
        
        # Sort timeline
        report["timeline"].sort(key=lambda x: x["timestamp"])
        
        return report


# Decorator for automatic node logging
def log_node_execution(node_name: str):
    """
    Decorator to automatically log node executions.
    
    Usage:
        @log_node_execution("my_node")
        def my_node(state: State) -> State:
            # node logic
            return state
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            import time
            
            logger_instance = NodeExecutionLogger()
            conversation_id = state.get("conversation_id", "unknown")
            
            # Capture input state
            input_state = dict(state)
            start_time = time.time()
            error = None
            
            try:
                # Execute node
                result = func(state, *args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                # Log execution
                execution_time = time.time() - start_time
                output_state = result if error is None else state
                
                logger_instance.log_node_execution(
                    conversation_id=conversation_id,
                    node_name=node_name,
                    input_state=input_state,
                    output_state=output_state,
                    execution_time=execution_time,
                    error=error
                )
        
        return wrapper
    return decorator


# Example usage in your nodes:
"""
from node_logger import log_node_execution

@log_node_execution("analyze_query")
def analyze_query_node(state: FamilyLawState) -> FamilyLawState:
    # Your existing logic
    agent = QueryAnalyzer()
    response = agent.analyze_query(state)
    
    state["user_intent"] = response.get("user_intent")
    state["info_needed_list"] = response.get("info_needed_list", [])
    # ... rest of your code
    
    return state
"""