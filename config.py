import json
import os

# OpenAI Configuration
API_KEY_FILE = "openai.key"
MODEL_NAME = "gpt-5-nano-2025-08-07"

# Logging Configuration
LOG_DIRECTORY = "logs"

# Prompt Configuration
PROMPTS_FILE = "prompts.json"
PROMPT_NAME = "indoguide"


def load_prompts():
    """Load prompts from the prompts repository"""
    prompts_path = os.path.join(os.path.dirname(__file__), PROMPTS_FILE)
    with open(prompts_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_prompt(name: str = PROMPT_NAME):
    """
    Get a specific prompt by name
    
    Args:
        name: Name of the prompt to retrieve
        
    Returns:
        The prompt text, or fallback if name not found
    """
    prompts = load_prompts()
    for prompt_obj in prompts:
        if prompt_obj["name"] == name:
            prompt = prompt_obj["prompt"]
            if isinstance(prompt, list):
                return " ".join(prompt)
            return prompt
    
    # Fallback if specified prompt not found
    return "You are a helpful assistant."

SYSTEM_PROMPT = get_prompt(PROMPT_NAME)
