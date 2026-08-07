import os
import sys
import torch
import numpy as np
import random
import json
import argparse

# Path resolution for importing phase_i and phase_ii modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PHASE_I_DIR = os.path.join(BASE_DIR, 'phase_i')
PHASE_II_DIR = os.path.join(BASE_DIR, 'phase_ii')

if PHASE_I_DIR not in sys.path:
    sys.path.insert(0, PHASE_I_DIR)
if PHASE_II_DIR not in sys.path:
    sys.path.insert(0, PHASE_II_DIR)

from gnn_model import BipartiteGNN
from data_generator import generate_company_scenario
from train import BipartiteGraphDataset
from extractor import RuleBasedExtractor, NLPExtractor, reconstruct_scenario

def print_separator(char='-', length=80):
    print(char * length)

DEFAULT_SAMPLE_PROMPT = (
    "Factories Alpha, Delta and Echo supplied the store in New York City. "
    "A recent inspection revealed that only 80% of the sample was free of defects. "
    "Similarly, factories Bravo, Charlie and Delta supplied the store in Los Angeles, "
    "which had a pass rate of only 70%. However, there was some better news from Chicago, "
    "Houston and Phoenix, which each had over 95% success rates. "
    "Chicago was given products from factories Alpha, Bravo and Charlie. "
    "Factories Bravo Delta and Echo sent their shipments to Houston. "
    "Lastly, Phoenix received products from factories Alpha, Echo and Charlie."
)

def evaluate_prompt_text(text, store_names=None, factory_names=None, model=None, threshold=0.25, device='cpu'):
    if store_names is None:
        store_names = ["New York City", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    if factory_names is None:
        factory_names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]

    print("\n" + "=" * 40)
    print(" NATURAL LANGUAGE REPORT TEXT PROCESSING")
    print("=" * 40)
    print(f"\nReport Text:\n{text.strip()}\n")

    extractor = RuleBasedExtractor()
    extracted_dict = extractor.extract(text, store_names, factory_names)

    print("--- Extracted Bipartite Connections & Store Outcomes ---")
    outcomes = extracted_dict.get('outcomes', {})
    for store in store_names:
        rate = outcomes.get(store, 0.95)
        status = "PASS" if rate >= 0.95 else "FAIL"
        print(f"Store {store:<15} | Status: {status:<5} | Success Rate: {rate:.2%}")

    print("\nConnections:")
    connections = extracted_dict.get('connections', [])
    for conn in connections:
        print(f"  • Factory {conn['entity']:<8} -> Store {conn['event']:<15} ({conn['units']} units)")

    dummy_scenario = {
        'num_factories': len(factory_names),
        'num_stores': len(store_names),
        'factory_names': factory_names,
        'store_names': store_names,
        'true_qualities': np.ones(len(factory_names), dtype=np.float32),
        'is_failed_factory': np.zeros(len(factory_names), dtype=np.int32),
        'sample_size': 100
    }

    scenario = reconstruct_scenario(extracted_dict, dummy_scenario)
    dataset_helper = BipartiteGraphDataset([scenario])
    graph_tensors = dataset_helper[0]

    with torch.no_grad():
        est_q, fail_p = model(
            graph_tensors['x_fac'].to(device),
            graph_tensors['x_store'].to(device),
            graph_tensors['edge_index_F'].to(device),
            graph_tensors['edge_index_S'].to(device),
            graph_tensors['edge_features'].to(device)
        )

    pred_qualities = est_q.cpu().numpy()
    pred_fail_probs = fail_p.cpu().numpy()
    pred_fails = (pred_fail_probs >= threshold).astype(int)

    print("\n" + "=" * 40)
    print(" FROZEN GNN QUALITY CONTROL PREDICTIONS")
    print("=" * 40)
    print(f"\n{'Factory':<12} | {'Est Quality':<12} | {'Fail Prob':<13} | {'Predicted Status':<15}")
    print_separator('-', 80)

    for i, fac in enumerate(factory_names):
        p_q = pred_qualities[i]
        prob = pred_fail_probs[i]
        p_status = "FAIL (<95%)" if pred_fails[i] == 1 else "PASS (>=95%)"
        print(f"Factory {fac:<5} | {p_q:<12.2%} | {prob:<13.2%} | {p_status:<15}")

    print_separator('=', 80)
    return pred_qualities, pred_fail_probs, pred_fails

def main():
    parser = argparse.ArgumentParser(description="Single Scenario Quality Control GNN Simulator")
    parser.add_argument('--text', type=str, help="Natural language report prompt to evaluate", default=None)
    parser.add_argument('--sample-prompt', action='store_true', help="Evaluate the standard benchmark sample prompt")
    args = parser.parse_args()

    print_separator('=', 80)
    print("      SINGLE SCENARIO GNN EVALUATION SIMULATOR - FACTORY QUALITY CONTROL")
    print_separator('=', 80)

    # 1. Load the frozen trained GNN model from Phase I
    model_path = os.path.join(PHASE_I_DIR, 'gnn_model.pth')
    device = torch.device('cpu')

    if not os.path.exists(model_path):
        print(f"[!] Warning: Trained model weights '{model_path}' not found.")
        model = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
    else:
        print(f"[*] Loading frozen GNN model weights from '{model_path}'...")
        model = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # Load validation-tuned threshold from Phase I thresholds.json
    threshold_path = os.path.join(PHASE_I_DIR, 'thresholds.json')
    threshold = 0.25
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, 'r') as f:
                thresholds_dict = json.load(f)
                if 'GNN (Proposed)' in thresholds_dict:
                    threshold = thresholds_dict['GNN (Proposed)']
                    print(f"[*] Loaded validation-tuned GNN threshold from Phase I: {threshold:.2f}")
        except Exception as e:
            print(f"[*] Error reading thresholds.json ({e}), using default threshold: {threshold:.2f}")

    if args.sample_prompt:
        evaluate_prompt_text(DEFAULT_SAMPLE_PROMPT, model=model, threshold=threshold, device=device)
        return

    if args.text:
        evaluate_prompt_text(args.text, model=model, threshold=threshold, device=device)
        return

    # Interactive Menu
    print("\nSelect simulation mode:")
    print("1. Sample Prompt Text Evaluation (Default Sample Report)")
    print("2. Custom Text Prompt Evaluation")
    print("3. Synthetic Scenario Simulation")
    try:
        choice = input("Enter choice (1, 2, or 3, default: 1): ").strip()
    except (IOError, EOFError):
        choice = '1'

    if choice == '2':
        custom_text = input("Enter report text prompt:\n").strip()
        if custom_text:
            evaluate_prompt_text(custom_text, model=model, threshold=threshold, device=device)
            return
        else:
            print("[!] Empty prompt entered, using default sample prompt.")
            evaluate_prompt_text(DEFAULT_SAMPLE_PROMPT, model=model, threshold=threshold, device=device)
            return
    elif choice == '3':
        num_factories = random.randint(5, 20)
        num_fails = max(1, int(round(num_factories * 0.20)))
        num_stores = max(2, int(round(num_factories / 2.0)))
        scenario = generate_company_scenario(num_factories=num_factories, num_fails=num_fails, num_stores=num_stores, avg_factories_per_store=3.0)
        
        dataset_helper = BipartiteGraphDataset([scenario])
        graph_tensors = dataset_helper[0]
        with torch.no_grad():
            est_q, fail_p = model(graph_tensors['x_fac'], graph_tensors['x_store'], graph_tensors['edge_index_F'], graph_tensors['edge_index_S'], graph_tensors['edge_features'])
        
        pred_qualities = est_q.numpy()
        pred_fail_probs = fail_p.numpy()
        pred_fails = (pred_fail_probs >= threshold).astype(int)
        
        print("\n--- Synthetic Scenario Predictions ---")
        for i in range(num_factories):
            status = "FAIL" if pred_fails[i] == 1 else "PASS"
            print(f"Factory {i:<2} | Est Quality: {pred_qualities[i]:.2%} | Fail Prob: {pred_fail_probs[i]:.2%} | Status: {status}")
    else:
        evaluate_prompt_text(DEFAULT_SAMPLE_PROMPT, model=model, threshold=threshold, device=device)

if __name__ == '__main__':
    main()
