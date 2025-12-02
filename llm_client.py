import os
from openai import OpenAI
from typing import Generator, Optional


class LLMClient:
    def __init__(self, api_key_file: str = "openai.key", model: str = "gpt-5-nano-2025-08-07"):
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
    
    def chat_stream(
        self, 
        user_message: str, 
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, str]:
        """
        Send a message and stream the response using Responses API
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt for instruction
            
        Yields:
            Chunks of the response text as they arrive
            
        Returns:
            The response_id for maintaining conversation context
        """
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
        
        return response_id
    
    def reset_conversation(self):
        """Reset the conversation context"""
        self.previous_response_id = None
