import json
import os
from datetime import datetime
from typing import List, Dict, Any
import re


class DialogueLogger:
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the dialogue logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Create a new session log structure
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session log dictionary
        """
        session_start_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "session_id": session_id,
            "session_start_time": session_start_time,
            "turns": []
        }
    
    def add_turn(
        self, 
        session_log: Dict[str, Any], 
        role: str, 
        text: str, 
        timestamp: str
    ):
        """
        Add a turn to the session log
        
        Args:
            session_log: The session log dictionary
            role: 'user' or 'bot'
            text: The message text
            timestamp: ISO format timestamp
        """
        turn = {
            "role": role,
            "timestamp": timestamp,
            "text": text
        }
        session_log["turns"].append(turn)
        
        # Print to console
        print(f"[{timestamp}] {role.upper()}: {text}")
    
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
