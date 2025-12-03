import os
from openai import OpenAI
from typing import Generator, Optional, List, Dict
from config import API_KEY_FILE, MODEL_NAME


class LLMClient:
    def __init__(self, api_key_file: str = API_KEY_FILE, model: str = MODEL_NAME):
        """
        Initialize the OpenAI client with Responses API
        
        Args:
            api_key_file: Path to file containing OpenAI API key
            model: Model name to use
        """
        # Read API key from file
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.previous_response_id: Optional[str] = None
        self.messages: List[Dict[str, str]] = []
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history
        
        Args:
            content: The user's message content
        """
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation history
        
        Args:
            content: The assistant's message content
        """
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get the full conversation history
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.messages.copy()
    
    def get_last_message(self) -> Optional[Dict[str, str]]:
        """
        Get the last message in the conversation
        
        Returns:
            The last message dictionary or None if no messages
        """
        return self.messages[-1] if self.messages else None
    
    def chat_stream(
        self, 
        user_message: str, 
        system_prompt: Optional[str] = None,
        auto_add_messages: bool = True
    ) -> Generator[str, None, str]:
        """
        Send a message and stream the response using Responses API
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt for instruction
            auto_add_messages: If True, automatically add messages to history
            
        Yields:
            Chunks of the response text as they arrive
            
        Returns:
            The response_id for maintaining conversation context
        """
        # Add user message to history if auto_add_messages is enabled
        if auto_add_messages:
            self.add_user_message(user_message)
        
        # Prepare the request parameters
        params = {
            "model": self.model,
            "input": user_message,
            "stream": True
        }
        
        # Add system prompt as instruction if provided
        if system_prompt:
            params["instructions"] = system_prompt
        
        # Add previous response ID for conversation continuity
        if self.previous_response_id:
            params["previous_response_id"] = self.previous_response_id
        
        # Create streaming response
        stream = self.client.responses.create(**params)

        full_text = ""
        response_id = None
        
        # Process the stream
        for event in stream:
            if event.type == "response.output_text.delta":
                chunk = event.delta
                full_text += chunk
                yield chunk
            elif event.type == "response.completed":
                response_id = event.response.id
        
        if response_id:
            self.previous_response_id = response_id
        
        # Add assistant message to history if auto_add_messages is enabled
        if auto_add_messages:
            self.add_assistant_message(full_text)
        
        return response_id
    
    def reset_conversation(self):
        """Reset the conversation context and message history"""
        self.previous_response_id = None
        self.messages = []
