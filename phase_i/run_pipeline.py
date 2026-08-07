import os
import torch
import numpy as np
import pandas as pd
import time

from data_generator import generate_dataset
from train import train_gnn
from evaluate import evaluate_all, plot_and_save_results

def main():
    start_time = time.time()
    
    # 1. Configuration
    num_train = 2000
    num_val = 300
    num_test = 100
    epochs = 20
    batch_size = 64
    hidden_dim = 64
    num_layers = 3
    lr = 0.001
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Data Generation
    print("Step 1: Generating datasets...")
    np.random.seed(42)
    # Generate combined to ensure reproducible splits
    total_graphs = num_train + num_val + num_test
    all_scenarios = generate_dataset(total_graphs, seed=42)
    
    train_scs = all_scenarios[:num_train]
    val_scs = all_scenarios[num_train:num_train + num_val]
    test_scs = all_scenarios[num_train + num_val:]
    
    print(f"Generated {len(train_scs)} train, {len(val_scs)} validation, and {len(test_scs)} test graphs.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gnn_path = os.path.join(script_dir, 'gnn_model.pth')
    mlp_path = os.path.join(script_dir, 'mlp_model.pth')
    thresh_path = os.path.join(script_dir, 'thresholds.json')
    
    # 3. Model Training & Loading
    import sys
    if '--eval-only' in sys.argv:
        print("\nStep 2: Skipping training. Loading model checkpoints...")
        from gnn_model import BipartiteGNN, FactoryMLP
        
        model_gnn = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)
        model_gnn.load_state_dict(torch.load(gnn_path, map_location=device))
        model_gnn = model_gnn.to(device)
        
        model_mlp = FactoryMLP(input_dim=4, hidden_dim=64)
        model_mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        model_mlp = model_mlp.to(device)
    else:
        print("\nStep 2: Training models...")
        from train import train_gnn, train_mlp
        
        print("\n--- Training Proposed Bipartite GNN ---")
        model_gnn = train_gnn(train_scs, val_scs, hidden_dim, num_layers, epochs, batch_size, lr, device, save_path=gnn_path)
        
        print("\n--- Training MLP Baseline (with Direct and Neighborhood attributes) ---")
        model_mlp = train_mlp(train_scs, val_scs, epochs, batch_size, lr, device, save_path=mlp_path)
    
    # 4. Evaluation
    print("\nStep 3: Running evaluation on test set (with validation-tuned thresholds)...")
    
    models_dict = {
        'GNN (Proposed)': {'type': 'gnn', 'model': model_gnn, 'ablate_edge': False, 'ablate_store': False},
        'MLP Baseline (No Success Rate)': {'type': 'mlp', 'model': model_mlp},
    }
    
    metrics, y_true_q, y_true_fail, metadata = evaluate_all(test_scs, models_dict, val_scs, device)
    
    # Save optimal thresholds to JSON
    thresholds_dict = {name: float(m['Best-Threshold']) for name, m in metrics.items()}
    import json
    with open(thresh_path, 'w') as f:
        json.dump(thresholds_dict, f, indent=4)
    print(f"\nSaved validation-tuned optimal thresholds to '{thresh_path}'")
    
    # 5. Format and print results
    results_rows = []
    for name, m in metrics.items():
        results_rows.append({
            'Model': name,
            'Quality MAE': f"{m['MAE']:.4f}",
            'Quality MSE': f"{m['MSE']:.6f}",
            'F1-Score': f"{m['F1-Score']:.4f}",
            'Precision': f"{m['Precision']:.4f}",
            'Recall': f"{m['Recall']:.4f}",
            'Accuracy': f"{m['Accuracy']:.4f}",
            'ROC-AUC': f"{m['ROC-AUC']:.4f}",
            'PR-AUC': f"{m['PR-AUC']:.4f}",
            'Threshold': f"{m['Best-Threshold']:.2f}"
        })
        
    # Format as markdown table manually to avoid dependency on tabulate
    if results_rows:
        headers = list(results_rows[0].keys())
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in results_rows]
        markdown_table = "\n".join([header_line, sep_line] + body_lines)
    else:
        markdown_table = ""
        
    print("\n=== EVALUATION RESULTS ===")
    print(markdown_table)
    
    # Save table to file
    with open(os.path.join(output_dir, 'metrics_table.md'), 'w') as f:
        f.write("# Metrics Summary Table\n\n")
        f.write(markdown_table)
        f.write("\n")
        
    # 6. Plotting
    print("\nStep 4: Generating and saving comparison plots...")
    plot_and_save_results(metrics, y_true_q, y_true_fail, metadata, plots_dir)
    
    elapsed = time.time() - start_time
    print(f"\nPipeline successfully completed in {elapsed:.2f} seconds!")

if __name__ == '__main__':
    main()
