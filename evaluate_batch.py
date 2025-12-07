
import json
import os
import math
import glob
from statistics import mean
from typing import List, Dict, Any, Optional

from config import config
from core.llm_client import LLMClient

# Initialize LLM Client
llm_client = LLMClient(model=config.LLM_JUDGE_MODEL)

def load_batch_results(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_laaj_prompt(metric_name: str) -> str:
    """Retrieves the system prompt for the specified LAAJ metric."""
    return config.get_prompt(f"laaj_{metric_name}")

# --- Retrieval Metrics ---

def calculate_recall_at_k(retrieved: List[Dict], ground_truth: List[Dict], k: int = 4) -> float:
    if not ground_truth:
        return 0.0
    
    retrieved_ids = [str(r['id']) for r in retrieved[:k]]
    ground_truth_ids = {str(g['id']) for g in ground_truth}
    
    # "Measures whether the correct supporting snippet appears in the top-4."
    # If ANY relevant doc is found, we count it as a hit (1.0).
    # If we want standard Recall@K, it is hits / total_relevant.
    # Given the description "whether the correct supporting snippet appears", 
    # and usually there's 1 correct answer snippet or a set.
    # Most RAG benchmarks use standard Recall@K (hits/relevant).
    # But if there are multiple ground truth snippets, and we find just 1, 
    # "Measures whether the correct supporting snippet appears" might imply Binary Recall (Hit Rate).
    # I will stick to standard Recall@K = (relevant_retrieved / total_relevant).
    
    hits = sum(1 for rid in retrieved_ids if rid in ground_truth_ids)
    return hits / len(ground_truth_ids) if ground_truth_ids else 0.0

def calculate_mrr(retrieved: List[Dict], ground_truth: List[Dict]) -> float:
    if not ground_truth:
        return 0.0
        
    retrieved_ids = [str(r['id']) for r in retrieved]
    ground_truth_ids = {str(g['id']) for g in ground_truth}
    
    for rank, rid in enumerate(retrieved_ids):
        if rid in ground_truth_ids:
            return 1.0 / (rank + 1)
            
    return 0.0

def calculate_ndcg_at_k(retrieved: List[Dict], ground_truth: List[Dict], k: int = 4) -> float:
    if not ground_truth:
        return 0.0
        
    retrieved_ids = [str(r['id']) for r in retrieved[:k]]
    ground_truth_ids = {str(g['id']) for g in ground_truth}
    
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids):
        rel = 1.0 if rid in ground_truth_ids else 0.0
        dcg += rel / math.log2(i + 2)
        
    # IDCG (Ideal DCG)
    # Best possible ordering: all relevant docs at the top
    ideal_rels = [1.0] * min(len(ground_truth_ids), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

# --- LLM-as-a-Judge Metrics ---

def parse_llm_json(response: str) -> Dict[str, Any]:
    try:
        # Simple/naive cleanup for markdown code blocks
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except Exception as e:
        print(f"Error parsing JSON from LLM: {e}")
        print(f"Raw response: {response}")
        return {"rating": 0, "reason": "Failed to parse LLM response"}

def evaluate_single_metric(metric: str, turn: Dict, system_context: str) -> Dict[str, Any]:
    """
    Generic function to run one LAAJ metric evaluation.
    metrics: factuality, faithfulness, helpfulness, overall
    """
    # The prompt text contains the instructions + format placeholders.
    # We treat the entire thing as the USER message to the judge.
    prompt_template = get_laaj_prompt(metric)
    
    user_input = turn.get("user_input", "")
    system_response = turn.get("system_response", "")
    retrieved = turn.get("retrieved_snippets", [])
    
    # Format retrieved snippets
    snippets_text = "\n".join([f"[{s['id']}] {s.get('content', '')}" for s in retrieved])
    
    # The placeholders in the prompt string need to be filled.
    # Using safe formatting in case of other braces, but our prompts use {system_response} etc.
    
    formatted_prompt = prompt_template.format(
        system_response=system_response,
        user_input=user_input,
        retrieved_snippets=snippets_text
    )
    
    try:
        # Use simple chat method
        response_text = llm_client.chat(
            user_message=formatted_prompt,
            auto_add_messages=False 
        )
        return parse_llm_json(response_text)
    except Exception as e:
        print(f"Error calling LLM for {metric}: {e}")
        return {"rating": 0, "reason": f"LLM call failed: {e}"}

def process_batch_file(input_file: str):
    print(f"Processing {input_file}...")
    data = load_batch_results(input_file)
    
    results_laaj = []
    
    total_recall = []
    total_mrr = []
    total_ndcg = []
    
    eval_metrics = {
        "factuality": [],
        "faithfulness": [],
        "helpfulness": [],
        "overall": []
    }
    
    for dialog in data.get("results", []):
        dialog_id = dialog.get("dialog_id")
        for turn in dialog.get("turns", []):
            turn_index = turn.get("turn_index")
            print(f"Evaluating Dialog {dialog_id}, Turn {turn_index}...")
            
            # Retrieval Metrics
            retrieved = turn.get("retrieved_snippets", [])
            ground_truth = turn.get("ground_truth_snippets", [])
            
            rec4 = calculate_recall_at_k(retrieved, ground_truth, k=4)
            mrr = calculate_mrr(retrieved, ground_truth)
            ndcg4 = calculate_ndcg_at_k(retrieved, ground_truth, k=4)
            
            total_recall.append(rec4)
            total_mrr.append(mrr)
            total_ndcg.append(ndcg4)
            
            # LAAJ Metrics
            turn_laaj = {
                "dialog_id": dialog_id,
                "turn_index": turn_index,
                "metrics": {}
            }
            
            for metric in ["factuality", "faithfulness", "helpfulness", "overall"]:
                res = evaluate_single_metric(metric, turn, "")
                turn_laaj["metrics"][metric] = res
                if res["rating"] > 0:
                    eval_metrics[metric].append(res["rating"])
            
            results_laaj.append(turn_laaj)
            
    # Aggregate
    metadata = data.get("metadata", {}).copy()
    metadata["eval_model"] = config.LLM_JUDGE_MODEL

    summary = {
        "metadata": metadata,
        "retrieval": {
            "mean_recall_at_4": mean(total_recall) if total_recall else 0,
            "mean_mrr": mean(total_mrr) if total_mrr else 0,
            "mean_ndcg_at_4": mean(total_ndcg) if total_ndcg else 0
        },
        "generation_quality": {
            k: mean(v) if v else 0 for k, v in eval_metrics.items()
        }
    }

    # Wrap LAAJ results with metadata
    laaj_output = {
        "metadata": metadata,
        "results": results_laaj
    }
    
    # Save Results
    os.makedirs(config.LAAJ_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    
    base_name = os.path.basename(input_file)
    # Remove prefix batchreplay_ if present
    clean_name = base_name.replace("batchreplay_", "")
    
    laaj_file = os.path.join(config.LAAJ_RESULTS_DIR, f"laaj_{clean_name}")
    eval_file = os.path.join(config.EVAL_RESULTS_DIR, f"eval_{clean_name}")
    
    with open(laaj_file, 'w') as f:
        json.dump(laaj_output, f, indent=2)
        
    with open(eval_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved LAAJ details to {laaj_file}")
    print(f"Saved Evaluation summary to {eval_file}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate batch replay results.")
    parser.add_argument("input_file", nargs="?", help="Path to the batch result JSON file to evaluate.")
    args = parser.parse_args()

    if args.input_file:
        if os.path.exists(args.input_file):
            process_batch_file(args.input_file)
        else:
            print(f"Error: File '{args.input_file}' not found.")
    else:
        # Find latest batch result
        files = glob.glob(os.path.join(config.BATCH_RESULTS_DIR, "*.json"))
        if not files:
            print("No batch results found to evaluate.")
        else:
            # Sort by modification time
            latest_file = max(files, key=os.path.getmtime)
            print(f"No input file specified. Using latest found: {latest_file}")
            process_batch_file(latest_file)
