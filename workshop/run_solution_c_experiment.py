import os
import torch
import numpy as np

from data_generator import generate_phase_ii_dataset, generate_dataset
from extractor import NLPExtractor, RuleBasedExtractor, reconstruct_scenario, compare_graphs
from evaluate import get_gnn_predictions
from gnn_model import BipartiteGNN
from train import BipartiteGraphDataset
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

def evaluate_gnn_on_scenarios(model, val_scs, test_scs, device='cpu'):
    # Validation threshold tuning
    val_subset = val_scs[:50]
    _, p_val = get_gnn_predictions(model, val_subset, device)
    y_val_fail = []
    for s in val_subset:
        y_val_fail.extend(s['is_failed_factory'])
    y_val_fail = np.array(y_val_fail)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_val_fail, p_val >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
            
    # Test set evaluation
    q_test, p_test = get_gnn_predictions(model, test_scs, device)
    y_test_fail = []
    for s in test_scs:
        y_test_fail.extend(s['is_failed_factory'])
    y_test_fail = np.array(y_test_fail)
    
    acc = accuracy_score(y_test_fail, (p_test >= best_t).astype(int))
    f1 = f1_score(y_test_fail, (p_test >= best_t).astype(int), zero_division=0)
    fpr, tpr, _ = roc_curve(y_test_fail, p_test)
    roc_auc = auc(fpr, tpr)
    return acc, f1, roc_auc, best_t

def main():
    device = torch.device('cpu')
    np.random.seed(42)
    
    print("=" * 85)
    print("   SOLUTION C (CROSS-ENCODER RELATION CLASSIFICATION) EVALUATION EXPERIMENTAL RUN")
    print("=" * 85)
    
    # 1. Load Phase I Validation Set for GNN Threshold Tuning
    num_train, num_val, num_test = 2000, 300, 100
    all_scenarios = generate_dataset(num_train + num_val + num_test, seed=42)
    val_scs = all_scenarios[num_train:num_train + num_val]
    
    # 2. Load Robust GNN Checkpoint
    print("\nLoading Robust Phase I GNN model ('gnn_model.pth')...")
    model_gnn = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
    model_gnn.load_state_dict(torch.load('gnn_model.pth', map_location=device))
    model_gnn = model_gnn.to(device)
    model_gnn.eval()
    
    # 3. Generate Phase II NLP Test Set
    print("\nGenerating Phase II report test scenarios (10 small reports for NLP evaluation)...")
    phase_ii_pairs = generate_phase_ii_dataset(150, seed=42)
    test_pairs = phase_ii_pairs[100:150]
    small_test_pairs = sorted(test_pairs, key=lambda x: x[0]['num_stores'])[:10]
    
    # 4. Extract with NLP Extractor featuring Solution C Cross-Encoder
    print("\nExtracting graphs with Upgraded NLPExtractor (Solution C Cross-Encoder)...")
    nlp_extractor = NLPExtractor(device="cpu")
    
    test_scs_recon_nlp = []
    metrics_nlp_list = []
    
    for idx, (s, r, c) in enumerate(small_test_pairs):
        ext = nlp_extractor.extract(r, s['store_names'], s['factory_names'])
        recon = reconstruct_scenario(ext, s)
        test_scs_recon_nlp.append(recon)
        cmp = compare_graphs(s, recon)
        metrics_nlp_list.append(cmp)
        
    avg_outcome_acc = np.mean([m['outcome_accuracy'] for m in metrics_nlp_list])
    avg_edge_prec = np.mean([m['edge_precision'] for m in metrics_nlp_list])
    avg_edge_rec = np.mean([m['edge_recall'] for m in metrics_nlp_list])
    avg_edge_f1 = np.mean([m['edge_f1'] for m in metrics_nlp_list])
    
    # 5. Downstream GNN Accuracy
    gnn_acc, gnn_f1, gnn_auc, thresh = evaluate_gnn_on_scenarios(model_gnn, val_scs, test_scs_recon_nlp, device)
    
    print("\n" + "=" * 85)
    print("                       EXPERIMENTAL RESULTS - SOLUTION C")
    print("=" * 85)
    print("--- Graph Extraction Accuracy ---")
    print(f"  - Store Outcome Accuracy: {avg_outcome_acc * 100:.2f}%")
    print(f"  - Edge Precision:       {avg_edge_prec * 100:.2f}%")
    print(f"  - Edge Recall:          {avg_edge_rec * 100:.2f}%")
    print(f"  - Edge Extraction F1:   {avg_edge_f1 * 100:.2f}%")
    print("\n--- Downstream GNN Quality Inference Accuracy ---")
    print(f"  - Downstream GNN Accuracy: {gnn_acc * 100:.2f}%")
    print(f"  - Downstream GNN F1-Score: {gnn_f1:.4f}")
    print(f"  - Downstream GNN ROC-AUC:  {gnn_auc:.4f}")

if __name__ == '__main__':
    main()
