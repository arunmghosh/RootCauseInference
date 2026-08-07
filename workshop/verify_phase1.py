import os
import torch
import numpy as np
import concurrent.futures
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, mean_absolute_error

from data_generator import generate_dataset
from evaluate import get_gnn_predictions, get_mlp_predictions, run_naive_frequency, run_failure_only_frequency, run_constraint_elimination
from gnn_model import BipartiteGNN, FactoryMLP
from baselines import run_bayesian_inference

def eval_single_bayes(scenario):
    return run_bayesian_inference(scenario, num_iter=150, burn_in=50)

def main():
    device = torch.device('cpu')
    np.random.seed(42)
    
    num_train, num_val, num_test = 2000, 300, 100
    total_graphs = num_train + num_val + num_test
    print(f"Generating Phase I dataset ({num_test} test graphs)...")
    all_scenarios = generate_dataset(total_graphs, seed=42)

    val_scs = all_scenarios[num_train:num_train + num_val]
    test_scs = all_scenarios[num_train + num_val:]

    print("Loading model checkpoints...")
    model_gnn = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=3)
    model_gnn.load_state_dict(torch.load('gnn_model.pth', map_location=device))
    model_gnn.eval()

    model_mlp = FactoryMLP(input_dim=4, hidden_dim=64)
    model_mlp.load_state_dict(torch.load('mlp_model.pth', map_location=device))
    model_mlp.eval()

    # Targets
    y_true_q = []
    y_true_fail = []
    for s in test_scs:
        y_true_q.extend(s['true_qualities'])
        y_true_fail.extend(s['is_failed_factory'])
    y_true_q = np.array(y_true_q)
    y_true_fail = np.array(y_true_fail)

    # 1. GNN
    print("Evaluating GNN (Proposed)...")
    q_gnn, p_gnn = get_gnn_predictions(model_gnn, test_scs, device)
    val_subset = val_scs[:50]
    _, p_val_gnn = get_gnn_predictions(model_gnn, val_subset, device)
    y_val_fail = []
    for s in val_subset:
        y_val_fail.extend(s['is_failed_factory'])
    y_val_fail = np.array(y_val_fail)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_val_fail, p_val_gnn >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    acc_gnn = accuracy_score(y_true_fail, (p_gnn >= best_t).astype(int))
    f1_gnn = f1_score(y_true_fail, (p_gnn >= best_t).astype(int), zero_division=0)
    fpr, tpr, _ = roc_curve(y_true_fail, p_gnn)
    auc_gnn = auc(fpr, tpr)
    mae_gnn = mean_absolute_error(y_true_q, q_gnn)

    # 2. Bayesian Inference Baseline
    print("Evaluating Bayesian Inference Baseline (parallel across CPUs)...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(eval_single_bayes, test_scs))
    
    q_bayes_list = []
    p_bayes_list = []
    for q, p in results:
        q_bayes_list.extend(q)
        p_bayes_list.extend(p)
    q_bayes = np.array(q_bayes_list)
    p_bayes = np.array(p_bayes_list)

    t_bayes = 0.3
    acc_bayes = accuracy_score(y_true_fail, (p_bayes >= t_bayes).astype(int))
    f1_bayes = f1_score(y_true_fail, (p_bayes >= t_bayes).astype(int), zero_division=0)
    fpr, tpr, _ = roc_curve(y_true_fail, p_bayes)
    auc_bayes = auc(fpr, tpr)
    mae_bayes = mean_absolute_error(y_true_q, q_bayes)

    # 3. MLP Baseline
    print("Evaluating MLP Baseline...")
    q_mlp, p_mlp = get_mlp_predictions(model_mlp, test_scs, device)
    _, p_val_mlp = get_mlp_predictions(model_mlp, val_subset, device)
    best_f1_mlp, best_t_mlp = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_val_fail, p_val_mlp >= t, zero_division=0)
        if f1 > best_f1_mlp:
            best_f1_mlp, best_t_mlp = f1, t
    acc_mlp = accuracy_score(y_true_fail, (p_mlp >= best_t_mlp).astype(int))
    f1_mlp = f1_score(y_true_fail, (p_mlp >= best_t_mlp).astype(int), zero_division=0)

    # Print Summary Table
    print("\n=================== PHASE I VERIFICATION RESULTS ===================")
    print(f"{'Model':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'MAE':<10}")
    print("-" * 75)
    print(f"{'GNN (Proposed)':<30} | {acc_gnn*100:6.2f}%    | {f1_gnn:8.4f}   | {auc_gnn:8.4f}   | {mae_gnn:8.4f}")
    print(f"{'Bayesian Inference Baseline':<30} | {acc_bayes*100:6.2f}%    | {f1_bayes:8.4f}   | {auc_bayes:8.4f}   | {mae_bayes:8.4f}")
    print(f"{'MLP Baseline':<30} | {acc_mlp*100:6.2f}%    | {f1_mlp:8.4f}   | {'N/A':<8}  | {'N/A':<8}")
    print("=" * 75)
    
    print(f"\n1. Is GNN Phase I Accuracy >= 85%?: {acc_gnn >= 0.85} ({acc_gnn*100:.2f}%)")
    print(f"2. Does GNN Accuracy exceed Bayesian Baseline?: {acc_gnn > acc_bayes} ({acc_gnn*100:.2f}% vs {acc_bayes*100:.2f}%)")

if __name__ == '__main__':
    main()
