import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

from gnn_model import BipartiteGNN
from baselines import run_naive_frequency, run_failure_only_frequency

class BipartiteGraphDataset(Dataset):
    """
    Custom PyTorch Dataset to convert generated scenarios into graph tensors.
    """
    def __init__(self, scenarios, ablate_edge_features=False, ablate_store_outcomes=False):
        self.scenarios = scenarios
        self.ablate_edge_features = ablate_edge_features
        self.ablate_store_outcomes = ablate_store_outcomes
        self.prepared_data = [self._prepare_scenario(s) for s in scenarios]
        
    def _prepare_scenario(self, s):
        N = s['num_factories']
        M = s['num_stores']
        edges = s['edges']
        
        # 1. Compute naive frequency and failure-only frequency
        # These are injected as node features to give the GNN a strong inductive bias
        _, naive_freq = run_naive_frequency(s)
        _, fail_only_freq = run_failure_only_frequency(s)
        
        # Factory node features: [1.0, naive_freq, fail_only_freq]
        x_fac = np.zeros((N, 3), dtype=np.float32)
        x_fac[:, 0] = 1.0
        x_fac[:, 1] = naive_freq
        x_fac[:, 2] = fail_only_freq
        
        # Store node features: [outcome, observed_defect_rate, 1 - outcome]
        x_store = np.zeros((M, 3), dtype=np.float32)
        if self.ablate_store_outcomes:
            # Ablate store outcomes: make all stores look identical (pass = 1.0, defect = 0, fail = 0)
            x_store[:, 0] = 1.0
            x_store[:, 1] = 0.0
            x_store[:, 2] = 0.0
        else:
            x_store[:, 0] = s['store_outcomes']
            x_store[:, 1] = 1.0 - s['store_success_rates']
            x_store[:, 2] = 1.0 - s['store_outcomes']
        
        # Edge indices and features
        edge_index_F = []
        edge_index_S = []
        edge_features = []
        
        for edge in edges:
            fac = edge['factory_idx']
            store = edge['store_idx']
            n_samples = edge['num_samples']
            
            store_tot_samples = s['store_samples'][store]
            prop = n_samples / store_tot_samples if store_tot_samples > 0 else 0.0
            
            edge_index_F.append(fac)
            edge_index_S.append(store)
            if self.ablate_edge_features:
                edge_features.append([1.0])
            else:
                edge_features.append([prop, float(n_samples)])
            
        edge_index_F = np.array(edge_index_F, dtype=np.int64) if edge_index_F else np.empty(0, dtype=np.int64)
        edge_index_S = np.array(edge_index_S, dtype=np.int64) if edge_index_S else np.empty(0, dtype=np.int64)
        
        edge_in_dim = 1 if self.ablate_edge_features else 2
        edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.empty((0, edge_in_dim), dtype=np.float32)
        
        # Normalize edge samples feature
        if len(edge_features) > 0 and not self.ablate_edge_features:
            edge_features[:, 1] = edge_features[:, 1] / 200.0  # standard scale
            
        # Build adjacency structures for calculation
        store_to_factories = [[] for _ in range(M)]
        factory_to_stores = [[] for _ in range(N)]
        for edge in edges:
            fac = edge['factory_idx']
            store = edge['store_idx']
            store_to_factories[store].append(fac)
            factory_to_stores[fac].append(store)
            
        # 1. Direct factory attributes
        num_failed_stores = np.zeros(N, dtype=np.float32)
        num_passed_stores = np.zeros(N, dtype=np.float32)
        for i in range(N):
            for store_idx in factory_to_stores[i]:
                if s['store_outcomes'][store_idx] == 0:
                    num_failed_stores[i] += 1.0
                else:
                    num_passed_stores[i] += 1.0
                    
        # 2. Neighborhood attributes
        co_suppliers_count = np.zeros(N, dtype=np.float32)
        avg_suppliers_at_connected = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            connected_stores = factory_to_stores[i]
            if len(connected_stores) > 0:
                co_facs = set()
                for store_idx in connected_stores:
                    for other_fac in store_to_factories[store_idx]:
                        if other_fac != i:
                            co_facs.add(other_fac)
                co_suppliers_count[i] = len(co_facs)
                avg_suppliers_at_connected[i] = sum(len(store_to_factories[store_idx]) for store_idx in connected_stores) / len(connected_stores)
            else:
                co_suppliers_count[i] = 0.0
                avg_suppliers_at_connected[i] = 0.0
                
        # Combine into MLP features (shape N, 4)
        x_mlp = np.zeros((N, 4), dtype=np.float32)
        x_mlp[:, 0] = num_failed_stores
        x_mlp[:, 1] = num_passed_stores
        x_mlp[:, 2] = co_suppliers_count / 20.0
        x_mlp[:, 3] = avg_suppliers_at_connected / 15.0
        
        # Targets
        y_quality = s['true_qualities'].astype(np.float32)
        y_fail = s['is_failed_factory'].astype(np.float32)
        
        return {
            'x_fac': torch.tensor(x_fac),
            'x_store': torch.tensor(x_store),
            'edge_index_F': torch.tensor(edge_index_F),
            'edge_index_S': torch.tensor(edge_index_S),
            'edge_features': torch.tensor(edge_features),
            'x_mlp': torch.tensor(x_mlp),
            'y_quality': torch.tensor(y_quality),
            'y_fail': torch.tensor(y_fail)
        }
        
    def __len__(self):
        return len(self.scenarios)
        
    def __getitem__(self, idx):
        return self.prepared_data[idx]

def bipartite_collate_fn(batch):
    """
    Collate function to combine multiple variable-sized bipartite graphs
    into a single batch by offsetting node indices.
    """
    x_fac_list = []
    x_store_list = []
    edge_index_F_list = []
    edge_index_S_list = []
    edge_features_list = []
    x_mlp_list = []
    y_quality_list = []
    y_fail_list = []
    
    fac_offset = 0
    store_offset = 0
    
    for item in batch:
        num_facs = item['x_fac'].size(0)
        num_stores = item['x_store'].size(0)
        
        x_fac_list.append(item['x_fac'])
        x_store_list.append(item['x_store'])
        edge_features_list.append(item['edge_features'])
        x_mlp_list.append(item['x_mlp'])
        y_quality_list.append(item['y_quality'])
        y_fail_list.append(item['y_fail'])
        
        # Offset edge indices
        edge_index_F_list.append(item['edge_index_F'] + fac_offset)
        edge_index_S_list.append(item['edge_index_S'] + store_offset)
        
        fac_offset += num_facs
        store_offset += num_stores
        
    return {
        'x_fac': torch.cat(x_fac_list, dim=0),
        'x_store': torch.cat(x_store_list, dim=0),
        'edge_index_F': torch.cat(edge_index_F_list, dim=0),
        'edge_index_S': torch.cat(edge_index_S_list, dim=0),
        'edge_features': torch.cat(edge_features_list, dim=0),
        'x_mlp': torch.cat(x_mlp_list, dim=0),
        'y_quality': torch.cat(y_quality_list, dim=0),
        'y_fail': torch.cat(y_fail_list, dim=0)
    }

def train_gnn(train_scenarios, val_scenarios, hidden_dim=64, num_layers=3, epochs=20, batch_size=64, lr=0.001, device='cpu',
              ablate_edge_features=False, ablate_store_outcomes=False, ablate_connectivity=False, edge_dropout=0.0, save_path='gnn_model.pth'):
    """
    Trains the Bipartite GNN on the training dataset and saves the best model.
    Supports edge_dropout (random edge perturbation during training) for robustness.
    """
    print(f"Creating datasets (Train: {len(train_scenarios)}, Val: {len(val_scenarios)})")
    if edge_dropout > 0.0:
        print(f"Applying Edge Dropout rate = {edge_dropout:.2f} during training...")
    train_dataset = BipartiteGraphDataset(train_scenarios, ablate_edge_features=ablate_edge_features, ablate_store_outcomes=ablate_store_outcomes)
    val_dataset = BipartiteGraphDataset(val_scenarios, ablate_edge_features=ablate_edge_features, ablate_store_outcomes=ablate_store_outcomes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=bipartite_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=bipartite_collate_fn)
    
    edge_in_dim = 1 if ablate_edge_features else 2
    actual_num_layers = 0 if ablate_connectivity else num_layers
    
    model = BipartiteGNN(fac_in_dim=3, store_in_dim=3, edge_in_dim=edge_in_dim, hidden_dim=hidden_dim, num_layers=actual_num_layers)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss weights: we combine regression loss (MSE) and classification loss (BCE)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCELoss()
    
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_loss_reg = 0.0
        train_loss_cls = 0.0
        
        start_time = time.time()
        
        for batch in train_loader:
            x_fac = batch['x_fac'].to(device)
            x_store = batch['x_store'].to(device)
            edge_index_F = batch['edge_index_F'].to(device)
            edge_index_S = batch['edge_index_S'].to(device)
            edge_features = batch['edge_features'].to(device)
            y_quality = batch['y_quality'].to(device)
            y_fail = batch['y_fail'].to(device)
            
            optimizer.zero_grad()
            
            # Apply edge dropout during training if enabled
            if edge_dropout > 0.0:
                E = edge_index_F.size(0)
                keep_mask = (torch.rand(E, device=device) >= edge_dropout)
                if E > 0 and keep_mask.sum() == 0:
                    keep_mask[0] = True
                e_F = edge_index_F[keep_mask]
                e_S = edge_index_S[keep_mask]
                e_feat = edge_features[keep_mask]
            else:
                e_F, e_S, e_feat = edge_index_F, edge_index_S, edge_features
                
            est_q, fail_p = model(x_fac, x_store, e_F, e_S, e_feat)
            
            loss_reg = criterion_reg(est_q, y_quality)
            loss_cls = criterion_cls(fail_p, y_fail)
            loss = loss_reg + 1.0 * loss_cls
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_quality)
            train_loss_reg += loss_reg.item() * len(y_quality)
            train_loss_cls += loss_cls.item() * len(y_quality)
            
        total_train_factories = sum(s['num_factories'] for s in train_scenarios)
        train_loss /= total_train_factories
        train_loss_reg /= total_train_factories
        train_loss_cls /= total_train_factories
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_reg = 0.0
        val_loss_cls = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x_fac = batch['x_fac'].to(device)
                x_store = batch['x_store'].to(device)
                edge_index_F = batch['edge_index_F'].to(device)
                edge_index_S = batch['edge_index_S'].to(device)
                edge_features = batch['edge_features'].to(device)
                y_quality = batch['y_quality'].to(device)
                y_fail = batch['y_fail'].to(device)
                
                est_q, fail_p = model(x_fac, x_store, edge_index_F, edge_index_S, edge_features)
                
                loss_reg = criterion_reg(est_q, y_quality)
                loss_cls = criterion_cls(fail_p, y_fail)
                loss = loss_reg + 1.0 * loss_cls
                
                val_loss += loss.item() * len(y_quality)
                val_loss_reg += loss_reg.item() * len(y_quality)
                val_loss_cls += loss_cls.item() * len(y_quality)
                
        total_val_factories = sum(s['num_factories'] for s in val_scenarios)
        val_loss /= total_val_factories
        val_loss_reg /= total_val_factories
        val_loss_cls /= total_val_factories
        
        epoch_time = time.time() - start_time
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Time: {epoch_time:.2f}s | "
                  f"Train Loss: {train_loss:.4f} (MSE: {train_loss_reg:.4f}, BCE: {train_loss_cls:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (MSE: {val_loss_reg:.4f}, BCE: {val_loss_cls:.4f})")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(save_path))
    return model

from gnn_model import FactoryMLP

def train_mlp(train_scenarios, val_scenarios, epochs=20, batch_size=64, lr=0.001, device='cpu', save_path='mlp_model.pth'):
    """
    Trains the FactoryMLP baseline on local factory features.
    """
    print(f"Creating datasets for MLP (Train: {len(train_scenarios)}, Val: {len(val_scenarios)})")
    train_dataset = BipartiteGraphDataset(train_scenarios)
    val_dataset = BipartiteGraphDataset(val_scenarios)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=bipartite_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=bipartite_collate_fn)
    
    model = FactoryMLP(input_dim=4, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCELoss()
    
    best_val_loss = float('inf')
    
    total_train_factories = sum(s['num_factories'] for s in train_scenarios)
    total_val_factories = sum(s['num_factories'] for s in val_scenarios)
    
    print("Starting MLP training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_mlp = batch['x_mlp'].to(device)
            y_quality = batch['y_quality'].to(device)
            y_fail = batch['y_fail'].to(device)
            
            optimizer.zero_grad()
            est_q, fail_p = model(x_mlp)
            
            loss_reg = criterion_reg(est_q, y_quality)
            loss_cls = criterion_cls(fail_p, y_fail)
            loss = loss_reg + 1.0 * loss_cls
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_quality)
            
        train_loss /= total_train_factories
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_reg = 0.0
        val_loss_cls = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x_mlp = batch['x_mlp'].to(device)
                y_quality = batch['y_quality'].to(device)
                y_fail = batch['y_fail'].to(device)
                
                est_q, fail_p = model(x_mlp)
                
                loss_reg = criterion_reg(est_q, y_quality)
                loss_cls = criterion_cls(fail_p, y_fail)
                loss = loss_reg + 1.0 * loss_cls
                
                val_loss += loss.item() * len(y_quality)
                val_loss_reg += loss_reg.item() * len(y_quality)
                val_loss_cls += loss_cls.item() * len(y_quality)
                
        val_loss /= total_val_factories
        val_loss_reg /= total_val_factories
        val_loss_cls /= total_val_factories
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"MLP Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} (MSE: {val_loss_reg:.4f}, BCE: {val_loss_cls:.4f})")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
    print(f"MLP training completed. Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(save_path))
    return model

if __name__ == '__main__':
    # Test dataset creation and a single batch step
    from data_generator import generate_dataset
    print("Generating test data...")
    scenarios = generate_dataset(20)
    train_scs = scenarios[:15]
    val_scs = scenarios[15:]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_gnn(train_scs, val_scs, epochs=5, batch_size=4, device=device)
    print("Successfully ran test training loop!")
