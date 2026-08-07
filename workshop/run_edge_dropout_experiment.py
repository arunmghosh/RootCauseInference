import os
import torch
import numpy as np

from data_generator import generate_dataset, generate_phase_ii_dataset
from train import train_gnn
from evaluate import get_gnn_predictions
from gnn_model import BipartiteGNN
from extractor import RuleBasedExtractor, NLPExtractor, reconstruct_scenario
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

def evaluate_gnn_model(model, val_scs, test_scs, device='cpu'):
    # Get val predictions for threshold tuning
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
            
    # Test evaluation
    q_test, p_test = get_gnn_predictions(model, test_scs, device)
    y_test_fail = []
    for s in test_scs:
        y_test_fail.extend(s['is_failed_factory'])
    y_test_fail = np.array(y_test_fail)
    
    acc = accuracy_score(y_test_fail, (p_test >= best_t).astype(int))
    f1 = f1_score(y_test_fail, (p_test >= best_t).astype(int), zero_division=0)
    fpr, tpr, _ = roc_curve(y_test_fail, p_test)
    roc_auc = auc(fpr, tpr)
    
    return acc, f1, roc_auc

def main():
    device = torch.device('cpu')
    np.random.seed(42)
    
    print("=" * 80)
    print("   EDGE DROPOUT ROBUSTNESS EXPERIMENT (SOLUTION E)")
    print("=" * 80)
    
    # 1. Generate Phase I Data (Clean Graphs)
    num_train, num_val, num_test = 2000, 300, 100
    all_scenarios = generate_dataset(num_train + num_val + num_test, seed=42)
    train_scs = all_scenarios[:num_train]
    val_scs = all_scenarios[num_train:num_train + num_val]
    test_scs_clean = all_scenarios[num_train + num_val:]
    
    # 2. Load / Train Standard GNN (0% Edge Dropout)
    print("\n--- Model 1: Standard GNN (0% Edge Dropout) ---")
    model_std = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
    if os.path.exists('gnn_model.pth'):
        model_std.load_state_dict(torch.load('gnn_model.pth', map_location=device))
        model_std = model_std.to(device)
        model_std.eval()
    else:
        model_std = train_gnn(train_scs, val_scs, hidden_dim=64, num_layers=3, epochs=15, edge_dropout=0.0, save_path='gnn_model.pth', device=device)
        
    # 3. Train Robust GNN with 15% Edge Dropout
    print("\n--- Model 2: Robust GNN (15% Edge Dropout during Phase I training) ---")
    model_drop15 = train_gnn(train_scs, val_scs, hidden_dim=64, num_layers=3, epochs=20, edge_dropout=0.15, save_path='gnn_model_drop15.pth', device=device)
    
    # 4. Train Robust GNN with 25% Edge Dropout
    print("\n--- Model 3: Robust GNN (25% Edge Dropout during Phase I training) ---")
    model_drop25 = train_gnn(train_scs, val_scs, hidden_dim=64, num_layers=3, epochs=20, edge_dropout=0.25, save_path='gnn_model_drop25.pth', device=device)
    
    # 5. Generate Phase II Reconstructed Test Set (NLP Extractor & Rule-Based Extractor)
    print("\nGenerating Phase II report test scenarios...")
    phase_ii_pairs = generate_phase_ii_dataset(150, seed=42)
    test_pairs = phase_ii_pairs[100:150]  # 50 test report pairs
    
    rb_extractor = RuleBasedExtractor()
    test_scs_recon_rb = []
    for s, r, c in test_pairs:
        extracted = rb_extractor.extract(r, s['store_names'], s['factory_names'])
        test_scs_recon_rb.append(reconstruct_scenario(extracted, s))
        
    print("Reconstructing NLP Extractor test scenarios (10 small reports)...")
    small_test_pairs = sorted(test_pairs, key=lambda x: x[0]['num_stores'])[:10]
    nlp_extractor = NLPExtractor(device="cpu")
    test_scs_recon_nlp = []
    for s, r, c in small_test_pairs:
        extracted = nlp_extractor.extract(r, s['store_names'], s['factory_names'])
        test_scs_recon_nlp.append(reconstruct_scenario(extracted, s))
        
    # 6. Evaluate all models across environments
    models = {
        'Standard GNN (0% Edge Dropout)': model_std,
        'Robust GNN (15% Edge Dropout)': model_drop15,
        'Robust GNN (25% Edge Dropout)': model_drop25,
    }
    
    print("\n" + "=" * 90)
    print("                       EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 90)
    
    for name, model in models.items():
        acc_clean, f1_clean, auc_clean = evaluate_gnn_model(model, val_scs, test_scs_clean, device)
        acc_rb, f1_rb, auc_rb = evaluate_gnn_model(model, val_scs, test_scs_recon_rb, device)
        acc_nlp, f1_nlp, auc_nlp = evaluate_gnn_model(model, val_scs, test_scs_recon_nlp, device)
        
        print(f"\nModel: {name}")
        print(f"  - Phase I Clean Test Set:      Accuracy = {acc_clean*100:6.2f}% | F1 = {f1_clean:.4f} | ROC-AUC = {auc_clean:.4f}")
        print(f"  - Phase II Rule-Based Graphs:  Accuracy = {acc_rb*100:6.2f}% | F1 = {f1_rb:.4f} | ROC-AUC = {auc_rb:.4f}")
        print(f"  - Phase II NLP Extracted Graphs: Accuracy = {acc_nlp*100:6.2f}% | F1 = {f1_nlp:.4f} | ROC-AUC = {auc_nlp:.4f}")
        
    print("\nExperiment completed successfully.")

if __name__ == '__main__':
    main()
