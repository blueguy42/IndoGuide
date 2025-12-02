import json
import uuid
from datetime import datetime
from llm_client import LLMClient
from logger import DialogueLogger


def run_batch_replay(
    prompts_file: str,
    system_prompt: str = "You are a helpful AI assistant.",
    output_session_id: str = None
):
    """
    Run a batch of test prompts through the chatbot
    
    Args:
        prompts_file: Path to JSON file containing test prompts
        system_prompt: System prompt to use for the session
        output_session_id: Optional custom session ID for output
    """
    # Load test prompts
    with open(prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        prompts = data.get("prompts", [])
    
    # Initialize components
    llm_client = LLMClient()
    logger = DialogueLogger()
    
    # Create session
    session_id = output_session_id or f"batch_{uuid.uuid4()}"
    session_log = logger.create_session(session_id)
    
    print(f"\n{'='*60}")
    print(f"Starting Batch Replay - Session: {session_id}")
    print(f"System Prompt: {system_prompt}")
    print(f"Total Prompts: {len(prompts)}")
    print(f"{'='*60}\n")
    
    # Process each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Turn {i}/{len(prompts)} ---")
        
        # User turn
        user_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        logger.add_turn(
            session_log,
            role="user",
            text=prompt,
            timestamp=user_timestamp
        )
        
        # Bot response
        full_response = ""
        for chunk in llm_client.chat_stream(
            user_message=prompt,
            system_prompt=system_prompt
        ):
            full_response += chunk
        
        bot_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        logger.add_turn(
            session_log,
            role="bot",
            text=full_response,
            timestamp=bot_timestamp
        )
        
        print()  # Add spacing between turns
    
    # Save session
    logger.save_session(session_log)
    
    print(f"\n{'='*60}")
    print(f"Batch Replay Complete!")
    print(f"Session saved to: logs/{session_id}.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch replay of test prompts")
    parser.add_argument(
        "prompts_file",
        help="Path to JSON file containing test prompts"
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help="System prompt to use"
    )
    parser.add_argument(
        "--session-id",
        help="Custom session ID for output (optional)"
    )
    
    args = parser.parse_args()
    
    run_batch_replay(
        prompts_file=args.prompts_file,
        system_prompt=args.system_prompt,
        output_session_id=args.session_id
    )
