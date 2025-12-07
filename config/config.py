import json
import os

API_KEY_FILE = "openai.key"
PROMPTS_FILE = "data/prompts.json"
KNOWLEDGE_BASE_FILE = "data/indonesia_knowledge_base.json"
CHROMA_DB_PATH = "./chroma_db"

# Model Configuration
MODEL_NAME = "gpt-5-nano-2025-08-07"
EMBEDDING_MODEL = "text-embedding-3-small"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_RERANKER_MODEL = "gpt-5-nano-2025-08-07"
LLM_JUDGE_MODEL = "gpt-4o-mini-2024-07-18"

# Prompt Configuration
# Prompt Configuration
PERSONAS = {
    "neutral": {"name": "Neutral (Baseline)", "prompt_key": "indoguide_neutral"},
    "friendly": {"name": "Friendly", "prompt_key": "indoguide_friendly"},
    "professional": {"name": "Professional", "prompt_key": "indoguide_professional"},
}

DEFAULT_PERSONA = "neutral"
PROMPT_NAME = PERSONAS[DEFAULT_PERSONA]["prompt_key"]

STARTER_MESSAGE = "Hello! I'm IndoGuide. I'd love to help plan your Indonesian adventure, from dream itineraries and visas to local culture. What's on your mind today?"

# Logging Configuration
LOG_DIR = "logs"

# Batch Replay Configuration
TEST_DIALOGUES_FILE = "data/dialogues/test_dialogues.json"
BATCH_RESULTS_DIR = "results/batch"
LAAJ_RESULTS_DIR = "results/laaj"
EVAL_RESULTS_DIR = "results/eval"

# RAG Configuration
TOP_K_RETRIEVAL = 10  # Number of candidates to retrieve
TOP_K_FINAL = 4       # Number of final snippets to use

RAG_CONFIGS = {
    1: {
        "name": "Baseline (No Reranking)",
        "cli_key": "baseline",
        "details": [
            "• Top-10 initial vector retrieval",
            "• Direct top-4 selection"
        ]
    },
    2: {
        "name": "Cross-Encoder Reranking",
        "cli_key": "crossencoder",
        "details": [
            "• Top-10 initial vector retrieval",
            "• Top-4 Cross-Encoder reranking"
        ]
    },
    3: {
        "name": "LLM Reranking",
        "cli_key": "llm",
        "details": [
            "• Top-10 initial vector retrieval",
            "• Top-4 LLM reranking"
        ]
    }
}

RAG_ID_TO_NAME = {k: v["name"] for k, v in RAG_CONFIGS.items()}
RAG_CLI_KEY_TO_ID = {v["cli_key"]: k for k, v in RAG_CONFIGS.items()}
RAG_NAME_TO_ID = {v["name"]: k for k, v in RAG_CONFIGS.items()}
RAG_ID_TO_DETAILS = {k: (v["name"], v["details"]) for k, v in RAG_CONFIGS.items()}

# Prompt Loader
def load_prompts():
    """Load prompts from the prompts repository"""
    prompts_path = os.path.join(os.path.dirname(__file__), "..", PROMPTS_FILE)
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

