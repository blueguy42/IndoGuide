#!/usr/bin/env python3
"""
CLI interface for IndoGuide
Provides an interactive command-line chat experience with the LLM
"""

import sys
import argparse
import uuid
from datetime import datetime
from llm_client import LLMClient
from logger import DialogueLogger
from rag_system import RAGSystem
from config import SYSTEM_PROMPT


class ChatCLI:
    def __init__(self, rag_config: str = "baseline"):
        """Initialize the CLI chat interface
        
        Args:
            rag_config: RAG configuration ('baseline', 'crossencoder', or 'llm')
        """
        self.llm_client = LLMClient()
        self.system_prompt = SYSTEM_PROMPT
        self.running = True
        self.rag_config_name = rag_config
        
        # Map config name to numeric value for RAGSystem
        config_map = {
            "baseline": 1,
            "crossencoder": 2,
            "llm": 3
        }
        self.rag_config = config_map[rag_config]
        
        # Initialize logger
        self.session_id = str(uuid.uuid4())
        self.logger = DialogueLogger()
        self.session_log = self.logger.create_session(self.session_id)
        
        # Initialize RAG system
        config_display = {
            "baseline": "Baseline (No Reranking)",
            "crossencoder": "Cross-Encoder Reranking",
            "llm": "LLM Reranking"
        }
        print(f"\nInitializing RAG system ({config_display[rag_config]})...")
        self.rag_system = RAGSystem(config=self.rag_config)
        print("RAG system ready.\n")
        
    def print_header(self):
        """Print the CLI header"""
        print("\n" + "="*60)
        print("IndoGuide - CLI Interface")
        print("="*60)
        print(f"\nSession ID: {self.session_id}")
        config_display = {
            "baseline": "Baseline (No Reranking)",
            "crossencoder": "Cross-Encoder Reranking",
            "llm": "LLM Reranking"
        }
        print(f"RAG Configuration: {config_display[self.rag_config_name]}")
        print("\nIndoGuide is your smart travel companion designed to make")
        print("exploring Indonesia effortless. Ask away information on must-see")
        print("destinations, visas, transportation, safety, and local etiquettes,")
        print("so you can travel with confidence. Whether you're planning your")
        print("itinerary or navigating on the go, IndoGuide helps you experience")
        print("Indonesia like a pro!")
        print("\nCommands:")
        print("  /reset    - Start a new conversation")
        print("  /history  - Show conversation history")
        print("  /config   - Show current RAG configuration")
        print("  /exit     - Exit the CLI")
        print("\nType your message and press Enter to chat.\n")
        print("="*60 + "\n")
    
    def print_message(self, role: str, content: str):
        """
        Print a formatted message
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        if role == "user":
            print(f"\nYou: {content}")
        elif role == "assistant":
            print(f"\nAssistant: {content}")
    
    def show_history(self):
        """Display the conversation history"""
        messages = self.llm_client.get_messages()
        
        if not messages:
            print("\nNo conversation history yet.\n")
            return
        
        print("\n" + "="*60)
        print("Conversation History")
        print("="*60)
        
        for i, msg in enumerate(messages, 1):
            role_name = "You" if msg["role"] == "user" else "Assistant"
            print(f"\n[{i}] {role_name}:")
            print(f"    {msg['content']}")
        
        print("\n" + "="*60 + "\n")
    
    def reset_conversation(self):
        """Reset the conversation"""
        # Save current session
        self.logger.save_session(self.session_log)
        
        # Create new session
        self.session_id = str(uuid.uuid4())
        self.session_log = self.logger.create_session(self.session_id)
        self.llm_client.reset_conversation()
        print(f"\nConversation reset. New session ID: {self.session_id}\n")
    
    def show_config(self):
        """Display current RAG configuration"""
        print("\n" + "="*60)
        print("Current RAG Configuration")
        print("="*60)
        
        config_info = {
            "baseline": ("Baseline (No Reranking)", ["• Top-10 initial vector retrieval", "• Direct top-4 selection"]),
            "crossencoder": ("Cross-Encoder Reranking", ["• Top-10 initial vector retrieval", "• Top-4 Cross-Encoder reranking"]),
            "llm": ("LLM Reranking", ["• Top-10 initial vector retrieval", "• Top-4 LLM reranking"])
        }
        
        config_name, details = config_info[self.rag_config_name]
        print(f"\nConfiguration: {config_name}")
        for detail in details:
            print(detail)
        
        print("\n" + "="*60 + "\n")
    
    def handle_command(self, command: str) -> bool:
        """
        Handle special commands
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled, False otherwise
        """
        command = command.lower().strip()
        
        if command == "/exit":
            print("\nSaving session and exiting...")
            self.logger.save_session(self.session_log)
            print("Goodbye!\n")
            self.running = False
            return True
        elif command == "/reset":
            self.reset_conversation()
            return True
        elif command == "/history":
            self.show_history()
            return True
        elif command == "/config":
            self.show_config()
            return True
        
        return False
    
    def chat(self):
        """Main chat loop"""
        self.print_header()
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    self.handle_command(user_input)
                    continue
                
                # Get user timestamp
                user_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                
                # Retrieve relevant context using RAG
                print("\nRetrieving relevant information...", end="", flush=True)
                retrieved_snippets = self.rag_system.retrieve(user_input)
                context = self.rag_system.format_context(retrieved_snippets)
                print(" Done.")
                
                # Inject context into system prompt
                augmented_prompt = context + "\n" + self.system_prompt
                
                # Log user turn with retrieved snippets
                self.logger.add_turn(
                    self.session_log,
                    speaker="user",
                    utterance=user_input,
                    timestamp=user_timestamp,
                    retrieved_snippets=retrieved_snippets
                )
                
                # Get and stream assistant response
                print("\nAssistant: ", end="", flush=True)
                
                full_response = ""
                for chunk in self.llm_client.chat_stream(
                    user_message=user_input,
                    system_prompt=augmented_prompt
                ):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                print("\n")  # New line after response
                
                # Get bot timestamp
                bot_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                
                # Log bot turn
                self.logger.add_turn(
                    self.session_log,
                    speaker="assistant",
                    utterance=full_response,
                    timestamp=bot_timestamp
                )
                
                # Auto-save session after each turn
                self.logger.save_session(self.session_log)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except EOFError:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                continue


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="IndoGuide CLI - Interactive travel assistant for Indonesia"
    )
    parser.add_argument(
        "--rag-config",
        type=str,
        choices=["baseline", "crossencoder", "llm"],
        default="baseline",
        help="RAG configuration: baseline=No reranking, crossencoder=Cross-Encoder reranking, llm=LLM reranking (default: baseline)"
    )
    
    args = parser.parse_args()
    
    cli = ChatCLI(rag_config=args.rag_config)
    cli.chat()


if __name__ == "__main__":
    main()
