import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error, mean_squared_error

from train import BipartiteGraphDataset, bipartite_collate_fn
from baselines import run_naive_frequency, run_failure_only_frequency, run_constraint_elimination, run_bayesian_inference

def get_gnn_predictions(model, scenarios, device='cpu', ablate_edge_features=False, ablate_store_outcomes=False):
    """
    Get GNN predictions for all scenarios.
    """
    dataset = BipartiteGraphDataset(scenarios, ablate_edge_features=ablate_edge_features, ablate_store_outcomes=ablate_store_outcomes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=bipartite_collate_fn)
    
    model.eval()
    all_est_q = []
    all_fail_p = []
    
    with torch.no_grad():
        for batch in loader:
            x_fac = batch['x_fac'].to(device)
            x_store = batch['x_store'].to(device)
            edge_index_F = batch['edge_index_F'].to(device)
            edge_index_S = batch['edge_index_S'].to(device)
            edge_features = batch['edge_features'].to(device)
            
            est_q, fail_p = model(x_fac, x_store, edge_index_F, edge_index_S, edge_features)
            
            all_est_q.append(est_q.cpu().numpy())
            all_fail_p.append(fail_p.cpu().numpy())
            
    return np.concatenate(all_est_q), np.concatenate(all_fail_p)

def get_mlp_predictions(model, scenarios, device='cpu'):
    """
    Get MLP predictions for all scenarios.
    """
    dataset = BipartiteGraphDataset(scenarios)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=bipartite_collate_fn)
    
    model.eval()
    all_est_q = []
    all_fail_p = []
    
    with torch.no_grad():
        for batch in loader:
            x_mlp = batch['x_mlp'].to(device)
            est_q, fail_p = model(x_mlp)
            
            all_est_q.append(est_q.cpu().numpy())
            all_fail_p.append(fail_p.cpu().numpy())
            
    return np.concatenate(all_est_q), np.concatenate(all_fail_p)

def get_predictions_for_scenarios(models_dict, scenarios, device='cpu'):
    """
    Computes quality and failure probability predictions for all models
    across a set of scenarios.
    """
    preds_dict = {}
    
    # Pre-extract true labels for simplicity
    y_true_q = []
    y_true_fail = []
    for s in scenarios:
        y_true_q.extend(s['true_qualities'])
        y_true_fail.extend(s['is_failed_factory'])
    y_true_q = np.array(y_true_q)
    y_true_fail = np.array(y_true_fail)
    
    for name, m_info in models_dict.items():
        m_type = m_info['type']
        
        pred_q = []
        pred_fail = []
        
        if m_type == 'baseline_A':
            for s in scenarios:
                q, p = run_naive_frequency(s)
                pred_q.extend(q)
                pred_fail.extend(p)
        elif m_type == 'baseline_B':
            for s in scenarios:
                q, p = run_failure_only_frequency(s)
                pred_q.extend(q)
                pred_fail.extend(p)
        elif m_type == 'baseline_C':
            for s in scenarios:
                q, p = run_constraint_elimination(s)
                pred_q.extend(q)
                pred_fail.extend(p)
        elif m_type == 'baseline_D':
            for s in scenarios:
                q, p = run_bayesian_inference(s, num_iter=200, burn_in=80)
                pred_q.extend(q)
                pred_fail.extend(p)
        elif m_type == 'gnn':
            model = m_info['model']
            ablate_edge = m_info.get('ablate_edge', False)
            ablate_store = m_info.get('ablate_store', False)
            q, p = get_gnn_predictions(model, scenarios, device, ablate_edge, ablate_store)
            pred_q = q
            pred_fail = p
        elif m_type == 'mlp':
            model = m_info['model']
            q, p = get_mlp_predictions(model, scenarios, device)
            pred_q = q
            pred_fail = p
            
        preds_dict[name] = (np.array(pred_q), np.array(pred_fail))
        
    return preds_dict, y_true_q, y_true_fail

def evaluate_all(test_scenarios, models_dict, val_scenarios, device='cpu'):
    """
    Computes predictions for GNN, MLP, and baselines on validation to optimize threshold,
    then evaluates on the test scenarios using validation-tuned thresholds.
    """
    # Filter out baseline_D for validation threshold tuning to save computation time
    val_subset = val_scenarios[:30]
    val_models_dict = {k: v for k, v in models_dict.items() if v['type'] != 'baseline_D'}
    val_preds, _, y_val_fail = get_predictions_for_scenarios(val_models_dict, val_subset, device)
    
    # Tune thresholds on validation
    best_thresholds = {}
    for name, (pred_q, pred_fail) in val_preds.items():
        if 'Constraint' in name or val_models_dict[name]['type'] == 'baseline_C':
            best_thresholds[name] = 0.5
        else:
            thresholds = np.linspace(0.01, 0.99, 99)
            best_f1 = 0.0
            best_t = 0.5
            for t in thresholds:
                f1 = f1_score(y_val_fail, pred_fail >= t, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds[name] = best_t
            
    # Set default threshold for baseline_D
    for name, m_info in models_dict.items():
        if m_info['type'] == 'baseline_D':
            best_thresholds[name] = 0.3
            
    # 2. Get test predictions and metadata
    print(f"Generating predictions on {len(test_scenarios)} test scenarios...")
    test_preds, y_true_q, y_true_fail = get_predictions_for_scenarios(models_dict, test_scenarios, device)
    
    # Calculate metadata for test scenarios (for bucket analysis)
    metadata = []
    idx_start = 0
    for scenario in test_scenarios:
        N = scenario['num_factories']
        edges = scenario['edges']
        avg_fac_store = len(edges) / scenario['num_stores'] if scenario['num_stores'] > 0 else 0
        metadata.append({
            'num_factories': N,
            'avg_factories_per_store': avg_fac_store,
            'indices': list(range(idx_start, idx_start + N))
        })
        idx_start += N
        
    # 3. Compute final metrics using the validation thresholds
    metrics_summary = {}
    for name, (pred_q, pred_fail) in test_preds.items():
        mae = mean_absolute_error(y_true_q, pred_q)
        mse = mean_squared_error(y_true_q, pred_q)
        
        # Classification metrics
        fpr, tpr, _ = roc_curve(y_true_fail, pred_fail)
        roc_auc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_fail, pred_fail)
        pr_auc = auc(recall_curve, precision_curve)
        
        t = best_thresholds[name]
        binary_pred = (pred_fail >= t).astype(int)
        
        f1 = f1_score(y_true_fail, binary_pred, zero_division=0)
        precision = precision_score(y_true_fail, binary_pred, zero_division=0)
        recall = recall_score(y_true_fail, binary_pred, zero_division=0)
        acc = accuracy_score(y_true_fail, binary_pred)
        
        metrics_summary[name] = {
            'MAE': mae,
            'MSE': mse,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': acc,
            'Best-Threshold': t,
            'raw_q': pred_q,
            'raw_fail': pred_fail
        }
        
    return metrics_summary, y_true_q, y_true_fail, metadata

def plot_and_save_results(metrics, y_true_q, y_true_fail, metadata, output_dir):
    """
    Generates comparison plots dynamically for whatever models are in the metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    for name, m in metrics.items():
        fpr, tpr, _ = roc_curve(y_true_fail, m['raw_fail'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Failure Detection')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for name, m in metrics.items():
        precision, recall, _ = precision_recall_curve(y_true_fail, m['raw_fail'])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Failure Detection')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=300)
    plt.close()
    
    # 3. Performance by Company Size (N)
    # Define sizes: Small (<= 25), Medium (26 - 100), Large (> 100)
    buckets = {
        'Small (N <= 25)': [],
        'Medium (25 < N <= 100)': [],
        'Large (N > 100)': []
    }
    
    for meta in metadata:
        n = meta['num_factories']
        indices = meta['indices']
        if n <= 25:
            buckets['Small (N <= 25)'].extend(indices)
        elif n <= 100:
            buckets['Medium (25 < N <= 100)'].extend(indices)
        else:
            buckets['Large (N > 100)'].extend(indices)
            
    bucket_names = list(buckets.keys())
    model_names = list(metrics.keys())
    
    # Quality MAE by size
    plt.figure(figsize=(10, 6))
    x = np.arange(len(bucket_names))
    width = 0.8 / len(model_names) if len(model_names) > 0 else 0.2
    
    for j, name in enumerate(model_names):
        mae_by_size = []
        for bucket in bucket_names:
            idxs = buckets[bucket]
            if len(idxs) == 0:
                mae_by_size.append(0.0)
            else:
                mae_by_size.append(mean_absolute_error(y_true_q[idxs], metrics[name]['raw_q'][idxs]))
        
        offset = (j - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, mae_by_size, width, label=name)
        
    plt.ylabel('Quality Estimate MAE (Lower is Better)')
    plt.title('Quality Estimate Error by Company Size')
    plt.xticks(x, bucket_names)
    plt.legend()
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_by_company_size.png'), dpi=300)
    plt.close()
    
    # 4. Classification F1 Score by Company Size
    plt.figure(figsize=(10, 6))
    for j, name in enumerate(model_names):
        f1_by_size = []
        for bucket in bucket_names:
            idxs = buckets[bucket]
            if len(idxs) == 0:
                f1_by_size.append(0.0)
            else:
                y_true_b = y_true_fail[idxs]
                pred_fail = metrics[name]['raw_fail'][idxs]
                t = metrics[name]['Best-Threshold']
                binary_pred = (pred_fail >= t).astype(int)
                f1_by_size.append(f1_score(y_true_b, binary_pred, zero_division=0))
                
        offset = (j - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, f1_by_size, width, label=name)
        
    plt.ylabel('Failure Detection F1-Score (Higher is Better)')
    plt.title('F1-Score by Company Size')
    plt.xticks(x, bucket_names)
    plt.legend()
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_by_company_size.png'), dpi=300)
    plt.close()
    
    print(f"Generated and saved evaluation plots to {output_dir}")
