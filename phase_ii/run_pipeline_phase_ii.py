import os
import sys
import torch
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, mean_absolute_error

# Add phase_i directory to sys.path to allow importing phase_i modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_I_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'phase_i'))
if PHASE_I_DIR not in sys.path:
    sys.path.insert(0, PHASE_I_DIR)

from data_generator import generate_phase_ii_dataset
from extractor import RuleBasedExtractor, NLPExtractor, reconstruct_scenario, compare_graphs
from train import train_gnn, train_mlp
from evaluate import evaluate_all, plot_and_save_results
from gnn_model import BipartiteGNN, FactoryMLP

def print_separator(char='-', length=80):
    print(char * length)

def main():
    start_time = time.time()
    
    # 1. Configuration
    if '--full' in sys.argv:
        num_train = 8000
        num_val = 1000
        num_test = 1000
        print("Using full Phase II dataset split (8000 train, 1000 val, 1000 test).")
    else:
        num_train = 1000
        num_val = 150
        num_test = 50
        print("Using standard dataset split (1000 train, 150 val, 50 test). Use '--full' to use the 10,000 scenario split.")
        
    epochs = 15
    batch_size = 64
    hidden_dim = 64
    num_layers = 3
    lr = 0.001
    
    output_dir = SCRIPT_DIR
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 2. Data Generation
    print("\n[Step 1] Generating Phase II datasets (scenarios + reports)...")
    total_graphs = num_train + num_val + num_test
    all_pairs = generate_phase_ii_dataset(total_graphs, seed=42)
    
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:num_train + num_val]
    test_pairs = all_pairs[num_train + num_val:]
    
    print(f"Generated {len(train_pairs)} train, {len(val_pairs)} val, and {len(test_pairs)} test report pairs.")
    
    # 3. Graph Reconstruction
    print("\n[Step 2] Reconstructing training and validation graphs using Rule-Based Extractor...")
    rb_extractor = RuleBasedExtractor()
    
    t0 = time.time()
    train_scs_recon = []
    for s, r, c in train_pairs:
        extracted = rb_extractor.extract(r, s['store_names'], s['factory_names'])
        train_scs_recon.append(reconstruct_scenario(extracted, s))
        
    val_scs_recon = []
    for s, r, c in val_pairs:
        extracted = rb_extractor.extract(r, s['store_names'], s['factory_names'])
        val_scs_recon.append(reconstruct_scenario(extracted, s))
        
    print(f"Reconstruction of train & val graphs completed in {time.time() - t0:.2f} seconds.")
    
    # 4. Load Frozen Phase I GNN and MLP Model Checkpoints (No training in Phase II)
    print("\n[Step 3] Loading frozen Phase I GNN ('gnn_model.pth') and MLP ('mlp_model.pth') checkpoints...")
    gnn_path = os.path.join(PHASE_I_DIR, 'gnn_model.pth')
    mlp_path = os.path.join(PHASE_I_DIR, 'mlp_model.pth')
    
    model_gnn = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)
    model_gnn.load_state_dict(torch.load(gnn_path, map_location=device))
    model_gnn = model_gnn.to(device)
    model_gnn.eval()
    
    model_mlp = FactoryMLP(input_dim=4, hidden_dim=64)
    model_mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    model_mlp = model_mlp.to(device)
    model_mlp.eval()
        
    # 5. Graph Extraction Evaluation & Reconstruct Test Sets
    print("\n[Step 4] Reconstructing and evaluating test reports using Rule-Based and NLP (NER/QA) extractors...")
    
    test_scs_recon_rb = []
    metrics_rb_list = []
    
    for s, r, c in test_pairs:
        ext_rb = rb_extractor.extract(r, s['store_names'], s['factory_names'])
        recon_rb = reconstruct_scenario(ext_rb, s)
        test_scs_recon_rb.append(recon_rb)
        cmp_rb = compare_graphs(s, recon_rb)
        metrics_rb_list.append(cmp_rb)
        
    small_test_pairs = sorted(test_pairs, key=lambda x: x[0]['num_stores'])[:10]
    print(f"Running NLP Extractor (NER + QA) comparison on the {len(small_test_pairs)} smallest test reports...")
    
    nlp_extractor = NLPExtractor(device="cpu")
    
    test_scs_recon_rb_small = []
    test_scs_recon_nlp_small = []
    
    metrics_rb_small_list = []
    metrics_nlp_small_list = []
    
    complexity_metrics = {
        'a': {'rb': [], 'nlp': []},
        'b': {'rb': [], 'nlp': []},
        'c': {'rb': [], 'nlp': []}
    }
    
    t_recon_start = time.time()
    for idx, (s, r, c) in enumerate(small_test_pairs):
        # Rules-based extraction
        ext_rb = rb_extractor.extract(r, s['store_names'], s['factory_names'])
        recon_rb = reconstruct_scenario(ext_rb, s)
        test_scs_recon_rb_small.append(recon_rb)
        cmp_rb = compare_graphs(s, recon_rb)
        metrics_rb_small_list.append(cmp_rb)
        complexity_metrics[c]['rb'].append(cmp_rb)
        
        # NLP extractor (NER/QA models)
        ext_nlp = nlp_extractor.extract(r, s['store_names'], s['factory_names'])
        recon_nlp = reconstruct_scenario(ext_nlp, s)
        test_scs_recon_nlp_small.append(recon_nlp)
        cmp_nlp = compare_graphs(s, recon_nlp)
        metrics_nlp_small_list.append(cmp_nlp)
        complexity_metrics[c]['nlp'].append(cmp_nlp)
        
    print(f"Extraction of NLP Extractor subset completed in {time.time() - t_recon_start:.2f} seconds.")
    
    def get_avg_metrics(metrics_list):
        if not metrics_list:
            return 0.0, 0.0, 0.0, 0.0
        acc = np.mean([m['outcome_accuracy'] for m in metrics_list])
        prec = np.mean([m['edge_precision'] for m in metrics_list])
        rec = np.mean([m['edge_recall'] for m in metrics_list])
        f1 = np.mean([m['edge_f1'] for m in metrics_list])
        return acc, prec, rec, f1

    acc_rb_sub, prec_rb_sub, rec_rb_sub, f1_rb_sub = get_avg_metrics(metrics_rb_small_list)
    acc_nlp_sub, prec_nlp_sub, rec_nlp_sub, f1_nlp_sub = get_avg_metrics(metrics_nlp_small_list)
    
    print("\n=== EXTRACTOR ACCURACY COMPARISON (On NLP Subset) ===")
    print(f"Rule-Based:    Outcome Acc={acc_rb_sub:.2%}, Edge Prec={prec_rb_sub:.2%}, Edge Rec={rec_rb_sub:.2%}, Edge F1={f1_rb_sub:.2%}")
    print(f"NLP Extractor: Outcome Acc={acc_nlp_sub:.2%}, Edge Prec={prec_nlp_sub:.2%}, Edge Rec={rec_nlp_sub:.2%}, Edge F1={f1_nlp_sub:.2%}")
    
    print("\n=== EXTRACTION ACCURACY BY COMPLEXITY (On NLP Subset) ===")
    complexity_rows = []
    for c in ['a', 'b', 'c']:
        acc_rb_c, prec_rb_c, rec_rb_c, f1_rb_c = get_avg_metrics(complexity_metrics[c]['rb'])
        acc_nlp_c, prec_nlp_c, rec_nlp_c, f1_nlp_c = get_avg_metrics(complexity_metrics[c]['nlp'])
        
        complexity_rows.append({
            'Complexity': f"Level {c.upper()}",
            'RB Outcome Acc': f"{acc_rb_c:.2%}",
            'RB Edge F1': f"{f1_rb_c:.2%}",
            'NLP Outcome Acc': f"{acc_nlp_c:.2%}",
            'NLP Edge F1': f"{f1_nlp_c:.2%}"
        })
        print(f"Level {c.upper()}:")
        print(f"  Rule-Based    -> Outcome Acc={acc_rb_c:.2%}, Edge F1={f1_rb_c:.2%}")
        print(f"  NLP Extractor -> Outcome Acc={acc_nlp_c:.2%}, Edge F1={f1_nlp_c:.2%}")
        
    headers_comp = ['Complexity', 'RB Outcome Acc', 'RB Edge F1', 'NLP Outcome Acc', 'NLP Edge F1']
    header_line_comp = "| " + " | ".join(headers_comp) + " |"
    sep_line_comp = "| " + " | ".join(["---"] * len(headers_comp)) + " |"
    body_lines_comp = ["| " + " | ".join(str(row[h]) for h in headers_comp) + " |" for row in complexity_rows]
    ext_table_markdown = "\n".join([header_line_comp, sep_line_comp] + body_lines_comp)
    
    with open(os.path.join(output_dir, 'phase_ii_extractor_comparison.md'), 'w') as f:
        f.write("# Extractor Accuracy & Complexity Analysis (Head-to-Head Subset)\n\n")
        f.write("### Subset Metrics\n")
        f.write(f"- **Rule-Based Extractor**: Outcome Accuracy: {acc_rb_sub:.2%}, Edge F1: {f1_rb_sub:.2%}\n")
        f.write(f"- **NLP Extractor (NER/QA)**: Outcome Accuracy: {acc_nlp_sub:.2%}, Edge F1: {f1_nlp_sub:.2%}\n\n")
        f.write("### Breakdown by Complexity Level\n")
        f.write(ext_table_markdown)
        f.write("\n")
        
    # 6. Downstream Predictions Evaluation (Including All Phase I Baselines)
    print("\n[Step 5] Evaluating Downstream quality control models and Phase I baseline bots...")
    
    models_dict = {
        'GNN (Proposed)': {'type': 'gnn', 'model': model_gnn, 'ablate_edge': False, 'ablate_store': False},
        'MLP Baseline': {'type': 'mlp', 'model': model_mlp},
        'Baseline A (Naive Freq)': {'type': 'baseline_A'},
        'Baseline B (Failure-Only Freq)': {'type': 'baseline_B'},
        'Baseline C (Constraint Elimination)': {'type': 'baseline_C'},
        'Baseline D (Bayesian Inference)': {'type': 'baseline_D'}
    }
    
    print("\n--- Running Evaluation on Full Rule-Based Reconstructed Test Set (50 graphs) ---")
    metrics_recon_rb, y_true_q, y_true_fail, metadata = evaluate_all(test_scs_recon_rb, models_dict, val_scs_recon, device)
    
    print("\n--- Running Evaluation on Small NLP Reconstructed Test Set (10 graphs) ---")
    metrics_recon_nlp_small, y_true_q_small, y_true_fail_small, metadata_small = evaluate_all(test_scs_recon_nlp_small, models_dict, val_scs_recon, device)
    
    print("\n--- Running Evaluation on Small Rule-Based Reconstructed Test Set (10 graphs) ---")
    metrics_recon_rb_small, _, _, _ = evaluate_all(test_scs_recon_rb_small, models_dict, val_scs_recon, device)
    
    def build_markdown_table(metrics_dict):
        results_rows = []
        for name, m in metrics_dict.items():
            results_rows.append({
                'Model': name,
                'Quality MAE': f"{m['MAE']:.4f}",
                'Quality MSE': f"{m['MSE']:.6f}",
                'F1-Score': f"{m['F1-Score']:.4f}",
                'Precision': f"{m['Precision']:.4f}",
                'Recall': f"{m['Recall']:.4f}",
                'Accuracy': f"{m['Accuracy']:.4f}",
                'ROC-AUC': f"{m['ROC-AUC']:.4f}",
                'PR-AUC': f"{m['PR-AUC']:.4f}"
            })
        headers = list(results_rows[0].keys())
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in results_rows]
        return "\n".join([header_line, sep_line] + body_lines)
        
    table_rb_full = build_markdown_table(metrics_recon_rb)
    table_nlp_small = build_markdown_table(metrics_recon_nlp_small)
    table_rb_small = build_markdown_table(metrics_recon_rb_small)
    
    print("\n=== DOWNSTREAM QUALITY CONTROL INFERENCE ON FULL RULE-BASED RECONSTRUCTED GRAPHS ===")
    print(table_rb_full)
    
    print("\n=== DOWNSTREAM QUALITY CONTROL INFERENCE ON SMALL NLP RECONSTRUCTED GRAPHS ===")
    print(table_nlp_small)
    
    # Save tables to output dir
    with open(os.path.join(output_dir, 'phase_ii_metrics_rb_full.md'), 'w') as f:
        f.write("# Downstream Inference Performance (On Full Rule-Based Reconstructed Graphs)\n\n")
        f.write(table_rb_full)
        f.write("\n")
        
    with open(os.path.join(output_dir, 'phase_ii_metrics_nlp_small.md'), 'w') as f:
        f.write("# Downstream Inference Performance (On Small NLP Reconstructed Graphs)\n\n")
        f.write(table_nlp_small)
        f.write("\n")
        
    with open(os.path.join(output_dir, 'phase_ii_metrics_rb_small.md'), 'w') as f:
        f.write("# Downstream Inference Performance (On Small Rule-Based Reconstructed Graphs)\n\n")
        f.write(table_rb_small)
        f.write("\n")
        
    plot_and_save_results(metrics_recon_rb, y_true_q, y_true_fail, metadata, plots_dir)
    
    elapsed = time.time() - start_time
    print(f"\nPhase II Pipeline completed in {elapsed:.2f} seconds!")

if __name__ == '__main__':
    main()
