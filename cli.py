#!/usr/bin/env python3
"""
CLI interface for IndoGuide
Provides an interactive command-line chat experience with the LLM
"""

import sys
from datetime import datetime
from llm_client import LLMClient
from config import SYSTEM_PROMPT


class ChatCLI:
    def __init__(self):
        """Initialize the CLI chat interface"""
        self.llm_client = LLMClient()
        self.system_prompt = SYSTEM_PROMPT
        self.running = True
        
    def print_header(self):
        """Print the CLI header"""
        print("\n" + "="*60)
        print("IndoGuide - CLI Interface")
        print("="*60)
        print("\nIndoGuide is your smart travel companion designed to make")
        print("exploring Indonesia effortless. Ask away information on must-see")
        print("destinations, visas, transportation, safety, and local etiquettes,")
        print("so you can travel with confidence. Whether you're planning your")
        print("itinerary or navigating on the go, IndoGuide helps you experience")
        print("Indonesia like a pro!")
        print("\nCommands:")
        print("  /reset    - Start a new conversation")
        print("  /history  - Show conversation history")
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
        self.llm_client.reset_conversation()
        print("\nConversation reset. Starting fresh!\n")
    
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
            print("\nGoodbye!\n")
            self.running = False
            return True
        elif command == "/reset":
            self.reset_conversation()
            return True
        elif command == "/history":
            self.show_history()
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
                
                # Get and stream assistant response
                print("\nAssistant: ", end="", flush=True)
                
                full_response = ""
                for chunk in self.llm_client.chat_stream(
                    user_message=user_input,
                    system_prompt=self.system_prompt
                ):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                print("\n")  # New line after response
                
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
    cli = ChatCLI()
    cli.chat()


if __name__ == "__main__":
    main()
