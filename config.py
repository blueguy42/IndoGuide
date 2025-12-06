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

# RAG Configuration
KNOWLEDGE_BASE_FILE = "indonesia_knowledge_base.json"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_RERANKER_MODEL = "gpt-5-nano-2025-08-07"
TOP_K_RETRIEVAL = 10  # Number of candidates to retrieve
TOP_K_FINAL = 4  # Number of final snippets to use


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
                return "\n".join(prompt)
            return prompt
    
    # Fallback if specified prompt not found
    return "You are a helpful assistant."

SYSTEM_PROMPT = get_prompt(PROMPT_NAME)

