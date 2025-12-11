# IndoGuide

## Table of Contents

- [Overview](#overview)
- [Interface Options](#interface-options)
- [System Configurations](#system-configurations)
  - [Persona Configurations](#persona-configurations-3-options)
  - [RAG Configurations](#rag-configurations-3-options)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Models Used](#models-used)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Batch Evaluation](#batch-evaluation)

## Overview

This repository is the technical code of an assignment for the Conversational AI module of the **MPhil in Human-Inspired Artificial Intelligence** programme at the **University of Cambridge**. This project implements **IndoGuide**, an intelligent travel companion chatbot designed to help users explore Indonesia by providing information on must-see destinations, visas, transportation, safety, and local etiquettes.

The system leverages Retrieval-Augmented Generation (RAG) with multiple reranking strategies to deliver contextually relevant and accurate responses.

## Interface Options

IndoGuide provides two ways to interact with the system:

1. **Web Interface (Streamlit)** - `app.py`

   - User-friendly, interactive web application
   - Session management with persistent conversation history
   - Real-time RAG configuration and persona selection

2. **Command Line Interface (CLI)** - `cli.py`
   - Terminal-based interaction
   - Ideal for scripted interactions and testing
   - Direct control over configuration parameters

## System Configurations

### Persona Configurations (3 Options)

The system supports three persona configurations that affect the tone and style of responses:

- **Neutral (Baseline)**: Standard, informative responses
- **Friendly**: Warm, conversational, and engaging tone
- **Professional**: Formal, detailed, and comprehensive responses

### RAG Configurations (3 Options)

The Retrieval-Augmented Generation system offers three reranking strategies:

1. **Baseline (No Reranking)**

   - Top-10 initial vector retrieval
   - Direct top-4 selection
   - Fastest option, baseline performance

2. **Cross-Encoder Reranking**

   - Top-10 initial vector retrieval
   - Top-4 Cross-Encoder reranking (MS MARCO-based)
   - Balanced speed and accuracy

3. **LLM Reranking**
   - Top-10 initial vector retrieval
   - Top-4 LLM reranking using GPT
   - Best accuracy, slower performance

## Project Structure

```
IndoGuide/
├── app.py                           # Streamlit web application
├── cli.py                           # CLI chat interface
├── batch_replay.py                  # Batch dialogue replay for evaluation
├── evaluate_batch.py                # Evaluation metrics calculator
├── config/
│   └── config.py                    # Configuration management
├── core/
│   ├── llm_client.py               # LLM API client
│   ├── rag_system.py               # RAG system implementation
│   └── logger.py                   # Session logging utilities
├── data/
│   ├── indonesia_knowledge_base.json  # Knowledge base for RAG
│   ├── test_dialogues.json         # Test dialogues for batch replay
│   └── prompts.json                # System prompts for personas and metrics
├── results/
│   ├── batch/                      # Batch replay output results
│   ├── laaj/                       # LLM-as-a-Judge rating results
│   └── eval/                       # Evaluation metric calculations
├── logs/                           # Session conversation logs
├── assets/
│   └── style.css                   # Streamlit UI styling
└── environment.yml                 # Conda environment specification
```

### Key Files

- **Knowledge Base**: `data/indonesia_knowledge_base.json` - Contains all the factual information about Indonesia used by the RAG system
- **Test Dialogues**: `data/test_dialogues.json` - Collection of test dialogues used for batch replay and technical research evaluation
- **Prompts**: `data/prompts.json` - System prompts for different personas and evaluation metrics (factuality, faithfulness, helpfulness, overall)

### Results

- **`results/batch/`** - Output from batch dialogue replay, containing system responses and metadata
- **`results/laaj/`** - LLM-as-a-Judge ratings for responses (factuality, faithfulness, helpfulness, overall quality)
- **`results/eval/`** - Calculated evaluation metrics (Recall@K, MRR, NDCG@K, and averaged LAAJ metrics)

## Requirements

### Python Version

- **Python 3.10** or higher

### Dependencies

The project requires the following Python libraries:

| Library                 | Version | Purpose                                |
| ----------------------- | ------- | -------------------------------------- |
| `streamlit`             | ≥1.51   | Web interface framework                |
| `openai`                | ≥2.8    | OpenAI API client for LLM interactions |
| `python-dotenv`         | Latest  | Environment variable management        |
| `chromadb`              | ≥0.4.0  | Vector database for RAG                |
| `sentence-transformers` | ≥2.2.0  | Embedding models and cross-encoders    |

All dependencies are automatically installed via the conda environment.

## Models Used

The system utilizes several specialized models for different components:

| Component                   | Model                                  | Provider     | Purpose                                                            |
| --------------------------- | -------------------------------------- | ------------ | ------------------------------------------------------------------ |
| **Chatbot**                 | `gpt-5-nano-2025-08-07`                | OpenAI       | Main conversational AI for generating responses                    |
| **Embedding**               | `text-embedding-3-small`               | OpenAI       | Text vectorization for semantic retrieval                          |
| **Cross-Encoder Reranking** | `cross-encoder/ms-marco-MiniLM-L6-v2` | Hugging Face | Re-ranks retrieved documents for relevance                         |
| **LLM Reranker**            | `gpt-5-nano-2025-08-07`                | OpenAI       | LLM-based re-ranking of retrieved documents                        |
| **LLM-as-a-Judge (LAAJ)**   | `gpt-4o-mini-2024-07-18`               | OpenAI       | Evaluates response quality (factuality, faithfulness, helpfulness) |

**Note:** The chatbot and LLM reranker use the same model (`gpt-5-nano-2025-08-07`) for consistency and cost efficiency.

## Setup Instructions

### 1. Environment Setup

Create and activate the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate IndoGuide
```

This will install all required dependencies including:

- OpenAI API client
- Streamlit
- Chroma vector database
- Cross-Encoder models

### 2. API Configuration

Create an `openai.key` file in the root directory with your OpenAI API key:

```bash
echo "your-openai-api-key-here" > openai.key
```

## Running the Application

### Option 1: Web Interface (Streamlit)

Run the interactive web application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**

- Select persona and RAG configuration from the sidebar
- Real-time conversation history
- Session management with option to save or create new chat

### Option 2: CLI Interface

Run the command-line chat interface:

```bash
python cli.py [OPTIONS]
```

**Arguments:**

- `--persona {neutral, friendly, professional}` (default: `neutral`)
  - Choose the persona for the assistant
- `--rag-config {baseline, crossencoder, llm}` (default: `baseline`)
  - Choose the RAG reranking strategy

**Example:**

```bash
python cli.py --persona friendly --rag-config llm
```

**CLI Commands:**

- `/reset` - Start a new conversation
- `/history` - Show conversation history
- `/config` - Show current configuration
- `/exit` - Exit the CLI

## Batch Evaluation

### Batch Replay

Replay a set of test dialogues and collect system responses:

```bash
python batch_replay.py [OPTIONS]
```

**Arguments:**

- `--persona {neutral, friendly, professional}` (default: `neutral`)
  - Persona for responses
- `--rag-config {baseline, crossencoder, llm}` (default: `baseline`)
  - RAG reranking strategy
- `--input-file PATH` (default: `data/test_dialogues.json`)
  - Path to test dialogues JSON file
- `--output-dir PATH` (default: `results/batch`)
  - Directory to save replay results

**Example:**

```bash
python batch_replay.py --persona friendly --rag-config llm --output-dir results/batch
```

Results are saved as JSON files with metadata and turn-by-turn dialogue data.

### LLM-as-a-Judge Evaluation

Rate batch replay results using LLM-as-a-Judge metrics:

```bash
python evaluate_batch.py [OPTIONS]
```

**Arguments:**

- `--batch-result PATH`
  - Path to batch replay result file (from `results/batch/`)
- `--knowledge-base PATH` (default: `data/indonesia_knowledge_base.json`)
  - Path to knowledge base JSON file
- `--output-dir PATH` (default: `results/laaj`)
  - Directory to save LAAJ ratings
- `--eval-dir PATH` (default: `results/eval`)
  - Directory to save calculated metrics

**Example:**

```bash
python evaluate_batch.py --batch-result results/batch/batchreplay_baseline_neutral_gpt-5-nano-2025-08-07_20251207185252.json
```

**Metrics Generated:**

The evaluation produces:

- **LAAJ Ratings** (LLM-as-a-Judge): Factuality, Faithfulness, Helpfulness, Overall Quality (saved in `results/laaj/`)
- **Retrieval Metrics**: Recall@K, MRR, NDCG@K (saved in `results/eval/`)
- **Aggregated Metrics**: Mean scores across all test dialogues

### Workflow Example

Here's a typical evaluation workflow:

1. **Run batch replay** with different configurations:

   ```bash
   python batch_replay.py --rag-config baseline --persona neutral
   python batch_replay.py --rag-config crossencoder --persona neutral
   python batch_replay.py --rag-config llm --persona neutral
   ```

2. **Evaluate with LLM-as-a-Judge**:

   ```bash
   python evaluate_batch.py --batch-result results/batch/batchreplay_baseline_neutral_*.json
   python evaluate_batch.py --batch-result results/batch/batchreplay_crossencoder_neutral_*.json
   python evaluate_batch.py --batch-result results/batch/batchreplay_llm_neutral_*.json
   ```

3. **Review results** in `results/batch/`, `results/laaj/`, and `results/eval/` directories
