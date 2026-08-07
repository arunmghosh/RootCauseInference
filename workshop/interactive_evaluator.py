import os
import sys
import torch
import numpy as np
import json
import re

from extractor import RuleBasedExtractor, NLPExtractor, reconstruct_scenario
from gnn_model import BipartiteGNN
from train import BipartiteGraphDataset

# Define candidate cities and adjectives to construct entity candidate lists
CITIES = ["Austin", "Boston", "Chicago", "Denver", "Seattle", "Dallas", "Houston", "Miami", "Atlanta", "Phoenix", "Omaha", "Detroit", "Portland", "Orlando", "Nashville", "Charlotte", "Columbus", "ElPaso", "Memphis", "Baltimore"]
ADJECTIVES = ["Apex", "Blue", "Crown", "Delta", "Echo", "Falcon", "Global", "Helix", "Iron", "Jade", "Krypton", "Lunar", "Matrix", "Nova", "Onyx", "Prime", "Quantum", "Rapid", "Summit", "Titan"]

def get_candidate_names():
    factories = []
    stores = []
    for city in CITIES:
        for adj in ADJECTIVES:
            factories.append(f"{city} {adj} Factory")
            stores.append(f"{city} {adj} Store")
    return factories, stores

def print_separator(char='=', length=80):
    print(char * length)

def main():
    print_separator('=', 80)
    print("                INTERACTIVE PHASE II GRAPH EXTRACTION & GNN EVALUATION")
    print_separator('=', 80)
    
    # 1. Load the GNN model
    model_path = 'gnn_model_phase_ii.pth'
    if not os.path.exists(model_path):
        model_path = 'gnn_model.pth'
        
    device = torch.device('cpu')
    if os.path.exists(model_path):
        print(f"[*] Loading GNN model from '{model_path}'...")
        model = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("[!] No trained GNN model weights found in workspace.")
        print("    Initializing an untrained GNN model for demo purposes.")
        model = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
        
    model.eval()
    
    # Generate candidate lists
    candidate_factories, candidate_stores = get_candidate_names()
    
    # Pre-instantiate extractors
    rb_extractor = RuleBasedExtractor()
    nlp_extractor = None  # Lazy loading because it takes a moment to load pipelines
    
    # Example report
    default_prompt = (
        "Logistics update. Initially, Miami Summit Factory was planned to supply 100 units to Boston Helix Store, "
        "but the final delivery consisted of 50 units. Also, Seattle Onyx Factory delivered 80 units to Seattle Onyx Store.\n"
        "The initial inspection for Boston Helix Store was reported as failed with a success rate of 90.0%. However, "
        "after re-inspecting the inventory, the final success rate was 96.0% and it officially passed."
    )
    
    print("\n[Default Prompt Example]:")
    print(default_prompt)
    print_separator('-', 80)
    
    while True:
        try:
            print("\nEnter your company report/prompt (or press Enter to use default example, type 'exit' to quit):")
            user_input = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                user_input.append(line)
            
            prompt = "\n".join(user_input).strip()
            if not prompt:
                prompt = default_prompt
            elif prompt.lower() == 'exit':
                break
                
            print("\nSelect Extractor:")
            print("1. Rule-Based Extractor")
            print("2. NLP Extractor (NER + QA Pipelines)")
            choice = input("Enter choice (1 or 2, default: 1): ").strip() or "1"
            
            if choice == "2":
                if nlp_extractor is None:
                    print("[*] Loading NLP pipelines (NER + QA)... Please wait...")
                    nlp_extractor = NLPExtractor(device="cpu")
                extractor = nlp_extractor
                extractor_name = "NLP Extractor"
            else:
                extractor = rb_extractor
                extractor_name = "Rule-Based Extractor"
                
            print(f"\n[*] Extracting graph data using {extractor_name}...")
            # Extract 4 components
            extracted_dict = extractor.extract(prompt, candidate_stores, candidate_factories)
            
            print("\n--- Extracted 4-Component Dictionary ---")
            print(json.dumps(extracted_dict, indent=2))
            
            if not extracted_dict['entities'] or not extracted_dict['events']:
                print("[!] Warning: No entities (factories) or events (stores) were extracted.")
                print("    Ensure the prompt mentions valid Factory/Store names using the City + Adjective structure.")
                continue
                
            # Create a mock true_scenario to map extracted names back to indexes for reconstruction
            true_scenario = {
                'num_factories': len(extracted_dict['entities']),
                'num_stores': len(extracted_dict['events']),
                'true_qualities': np.ones(len(extracted_dict['entities']), dtype=np.float32) * 0.96,
                'is_failed_factory': np.zeros(len(extracted_dict['entities']), dtype=np.float32),
                'factory_names': extracted_dict['entities'],
                'store_names': extracted_dict['events']
            }
            
            # Reconstruct scenario
            recon_scenario = reconstruct_scenario(extracted_dict, true_scenario)
            
            print("\n--- Reconstructed Bipartite Graph ---")
            print(f"Number of Factories: {recon_scenario['num_factories']}")
            print(f"Number of Stores:    {recon_scenario['num_stores']}")
            print("Edges:")
            for e in recon_scenario['edges']:
                f_name = recon_scenario['factory_names'][e['factory_idx']]
                s_name = recon_scenario['store_names'][e['store_idx']]
                print(f"  - {f_name} -> {s_name} ({e['num_samples']} units)")
                
            print("\nStore Outcomes:")
            for idx, s_name in enumerate(recon_scenario['store_names']):
                rate = recon_scenario['store_success_rates'][idx]
                outcome = "PASS" if recon_scenario['store_outcomes'][idx] == 1.0 else "FAIL"
                print(f"  - {s_name}: Success Rate = {rate:.2%}, Outcome = {outcome}")
                
            # 2. Run GNN model for downstream quality control estimation
            print("\n[*] Running downstream GNN evaluation on reconstructed graph...")
            dataset = BipartiteGraphDataset([recon_scenario])
            graph_tensors = dataset[0]
            
            with torch.no_grad():
                est_q, fail_p = model(
                    graph_tensors['x_fac'],
                    graph_tensors['x_store'],
                    graph_tensors['edge_index_F'],
                    graph_tensors['edge_index_S'],
                    graph_tensors['edge_features']
                )
                
            pred_q = est_q.numpy()
            fail_probs = fail_p.numpy()
            
            # Threshold loading
            threshold = 0.3
            if os.path.exists('thresholds.json'):
                try:
                    with open('thresholds.json', 'r') as f:
                        th_dict = json.load(f)
                        if 'GNN (Proposed)' in th_dict:
                            threshold = th_dict['GNN (Proposed)']
                except Exception:
                    pass
                    
            print(f"Using GNN decision threshold: {threshold:.2f}\n")
            
            print_separator('-', 80)
            print(f"{'Factory Name':<30} | {'Estimated Quality':<20} | {'GNN Fail Prob':<15} | {'Predicted Status':<10}")
            print_separator('-', 80)
            for idx, f_name in enumerate(recon_scenario['factory_names']):
                q_est = pred_q[idx]
                prob = fail_probs[idx]
                status = "FAIL" if prob >= threshold else "PASS"
                print(f"{f_name:<30} | {q_est:<20.2%} | {prob:<15.2%} | {status:<10}")
            print_separator('-', 80)
            
        except KeyboardInterrupt:
            print("\nExiting interactive evaluator.")
            break
        except Exception as e:
            print(f"[!] Error processing input: {e}")

if __name__ == '__main__':
    main()
