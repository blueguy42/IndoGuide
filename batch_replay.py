import json
import os
import argparse
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any

from llm_client import LLMClient
from rag_system import RAGSystem
from logger import DialogueLogger
from config import (
    SYSTEM_PROMPT, 
    MODEL_NAME, 
    RAG_CLI_KEY_TO_ID, 
    RAG_ID_TO_NAME, 
)


class BatchReplay:
    def __init__(self, rag_config_key: str = "baseline", input_file: str = "dialogues/test_dialogues.json"):
        """
        Initialize the batch replay system
        
        Args:
            rag_config_key: CLI key for RAG configuration (e.g., 'baseline', 'llm')
            input_file: Path to the input JSON file containing test dialogues
        """
        self.input_file = input_file
        self.rag_config_key = rag_config_key
        self.rag_config_id = RAG_CLI_KEY_TO_ID.get(rag_config_key, 1)
        self.config_name = RAG_ID_TO_NAME[self.rag_config_id]
        
        print(f"Initializing Batch Replay with RAG Configuration: {self.config_name}")
        
        # Initialize components
        self.llm_client = LLMClient(model=MODEL_NAME)
        self.rag_system = RAGSystem(config=self.rag_config_id)
        self.logger = DialogueLogger()
        
    def load_dialogues(self) -> List[Dict[str, Any]]:
        """Load dialogues from the input JSON file"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
            
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def run(self, output_dir: str = "results/batch"):
        """
        Run the batch replay on all loaded dialogues
        
        Args:
            output_dir: Directory to save the results
        """
        dialogues = self.load_dialogues()
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "metadata": {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "rag_config": self.config_name,
                "rag_config_key": self.rag_config_key,
                "model": MODEL_NAME,
                "input_file": self.input_file
            },
            "results": []
        }
        
        print(f"Starting replay of {len(dialogues)} dialogues...")
        
        for i, dialog in enumerate(dialogues, 1):
            dialog_id = dialog.get("dialog_id", i)
            print(f"Processing Dialogue {i}/{len(dialogues)} (ID: {dialog_id})...")
            
            # Reset conversation state for new dialogue
            self.llm_client.reset_conversation()
            
            # Prepare result structure for this dialogue
            dialog_result = {
                "dialog_id": dialog_id,
                "turns": []
            }
            
            # Start timer for whole dialogue
            dialog_start_time = time.time()
            
            # Process turns
            # We iterate through the provided turns.
            # If it's a USER turn, we feed it to the system and generate a response.
            # If the NEXT provided turn is an ASSISTANT turn, we treat it as ground truth for the previous user turn.
            
            conversation_history = dialog["turns"]
            
            # We iterate by index to look ahead for ground truth
            idx = 0
            while idx < len(conversation_history):
                turn = conversation_history[idx]
                speaker = turn.get("speaker")
                
                if speaker == "user":
                    user_utterance = turn.get("utterance")
                    
                    # 1. RAG Retrieval
                    retrieved_snippets = self.rag_system.retrieve(user_utterance)
                    context = self.rag_system.format_context(retrieved_snippets)
                    
                    # 2. Augment Prompt
                    augmented_prompt = context + "\n" + SYSTEM_PROMPT
                    
                    # 3. Generate Response
                    start_gen = time.time()
                    # We use chat() for synchronous full response, or chat_stream() and accumulate.
                    # Since this is batch, chat_stream loop is fine to simulate the real app flow, 
                    # but chat() is cleaner if we don't need token-by-token timing.
                    # Let's use chat_stream to match app.py behavior exactly including message history management.
                    
                    full_response = ""
                    # The app.py does: augmented_prompt = context + "\n" + SYSTEM_PROMPT
                    # And passes it as `system_prompt` to `chat_stream`.
                    
                    for chunk in self.llm_client.chat_stream(
                        user_message=user_utterance, 
                        system_prompt=augmented_prompt,
                        auto_add_messages=True # Ensure history is maintained
                    ):
                        full_response += chunk
                        
                    latency = time.time() - start_gen
                    
                    # 4. Record Results
                    turn_result = {
                        "turn_index": idx,
                        "user_input": user_utterance,
                        "system_response": full_response,
                        "retrieved_snippets": retrieved_snippets,
                        "latency_seconds": latency
                    }
                    
                    # Check if next turn is assistant (Ground Truth)
                    if idx + 1 < len(conversation_history):
                        next_turn = conversation_history[idx + 1]
                        if next_turn.get("speaker") == "assistant":
                            turn_result["ground_truth_response"] = next_turn.get("utterance_ref")
                            turn_result["ground_truth_snippets"] = next_turn.get("grounding_snippets")
                            idx += 1 # Skip next turn as we handled it
                            
                    dialog_result["turns"].append(turn_result)
                    
                idx += 1
            
            dialog_result["total_time_seconds"] = time.time() - dialog_start_time
            results["results"].append(dialog_result)
            
        # Save Report
        timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        
        # Sanitize model name: alphanumeric or -, else _
        import re
        sanitized_model = re.sub(r'[^a-zA-Z0-9-]', '_', MODEL_NAME)
        
        output_filename = f"batchreplay_{self.rag_config_key}_{sanitized_model}_{timestamp_str}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"\nBatch replay completed. Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run batch replay evaluation for IndoGuide")
    parser.add_argument(
        "--input", 
        type=str, 
        default="dialogues/test_dialogues.json", 
        help="Path to input JSON file with dialogues"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        choices=list(RAG_CLI_KEY_TO_ID.keys()),
        default="baseline",
        help="RAG configuration to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/batch",
        help="Directory to save output results"
    )
    
    args = parser.parse_args()
    
    try:
        replay = BatchReplay(rag_config_key=args.config, input_file=args.input)
        replay.run(output_dir=args.output_dir)
    except Exception as e:
        print(f"Error running batch replay: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
