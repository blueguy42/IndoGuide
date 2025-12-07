import json
import os
from datetime import datetime
from typing import List, Dict, Any
import re
from config import LOG_DIR, RAG_CONFIGS


class DialogueLogger:
    def __init__(self, log_dir: str = LOG_DIR):
        """
        Initialize the dialogue logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def create_session(self, session_id: str, rag_config: int = 1) -> Dict[str, Any]:
        """
        Create a new session log structure
        
        Args:
            session_id: Unique session identifier
            rag_config: RAG configuration number (1=Baseline, 2=Cross-Encoder, 3=LLM)
            
        Returns:
            Session log dictionary
        """
        session_start_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "session_id": session_id,
            "session_start_time": session_start_time,
            "rag_config": RAG_CONFIGS.get(rag_config, {}).get("cli_key", f"unknown_{rag_config}"),
            "turns": []
        }
    
    def add_turn(
        self, 
        session_log: Dict[str, Any], 
        speaker: str, 
        utterance: str, 
        timestamp: str,
        retrieved_snippets: List[Dict[str, Any]] = None
    ):
        """
        Add a turn to the session log
        
        Args:
            session_log: The session log dictionary
            speaker: 'user' or 'assistant'
            utterance: The message text
            timestamp: ISO format timestamp
            retrieved_snippets: Optional list of retrieved RAG snippets
        """
        turn = {
            "speaker": speaker,
            "timestamp": timestamp,
            "utterance": utterance
        }
        
        # Add retrieved snippets if provided
        if retrieved_snippets is not None:
            turn["retrieved_snippets"] = retrieved_snippets
        
        session_log["turns"].append(turn)
        
        # Print to console
        print(f"[{timestamp}] {speaker.upper()}: {utterance}")
    
    def save_session(self, session_log: Dict[str, Any]):
        """
        Save session log to JSON file with format: timestamp_sessionId.json
        
        Args:
            session_log: The session log dictionary to save
        """
        session_start_time = re.sub("[^0-9]", "", session_log.get("session_start_time", "unknown"))
        session_id = session_log["session_id"]
        filename = f"conversation_{session_start_time}_{session_id}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_log, f, indent=2, ensure_ascii=False)
