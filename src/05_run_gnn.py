#!/usr/bin/env python3
"""
Graph Neural Network (GIN) for CFTR Modulator Response Prediction
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.append('../utils')
from gnn_utils import (
    normalize_column_names, normalize_taxon_name,
    build_global_taxonomic_tree, create_pyg_dataset
)

# Configuration
DATA_FILE = "../data/simulated_microbiome.csv"
TAXA_FILE = "../data/taxa_table_grouped_renamed.csv"
OUTPUT_DIR = "results/gnn"

OUTCOME = "zscore_label"
NORMALIZATION = "zscore"

# Hyperparameters
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 4
EPOCHS = 100 
LR = 1e-3     
DROPOUT = 0.5
HIDDEN = 48
K_FOLDS = 13
EMBEDDING_DIM = 24
N_LAYERS = 2

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


# Load / prepare data
lung_data = pd.read_csv(DATA_FILE)
taxa_info = pd.read_csv(TAXA_FILE, sep=";")

lung_data = normalize_column_names(lung_data) 
taxa_info = taxa_info.drop_duplicates(subset=["Labels"], keep="first") #pq 

lung_m0 = lung_data[lung_data["Time"] == "M0"].copy().reset_index(drop=True)
lung_m12 = lung_data[lung_data["Time"] == "M12"].copy().reset_index(drop=True)

taxon_cols = [c for c in lung_m0.columns if c.startswith("b_")]

print(f"Taxa detected: {len(taxon_cols)}")

#Building the tree:
def preserve_label_as_genus(row):
    label = str(row["Labels"]).strip()
    if label.startswith("Family:") or label.startswith("Order:") or label.upper() == "OTHER":
        return label
    else:
        return row["Genus"]

taxa_info["Genus"] = taxa_info.apply(preserve_label_as_genus, axis=1)

original_names = [c[2:] for c in taxon_cols]
genus_df = pd.DataFrame({
    "Genus": original_names,
    "Genus_modified": [normalize_taxon_name(name) for name in original_names]
})

taxa_subset = taxa_info[["Labels", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]].copy()
hierarchy = genus_df.merge(taxa_subset, left_on="Genus", right_on="Labels", how="left")
hierarchy = hierarchy.rename(columns={"Genus_y": "Genus", "Genus_x": "Genus_input"})

global_tree, global_taxa2id, global_id2taxa = build_global_taxonomic_tree(hierarchy)
print(f"Global tree: {len(global_taxa2id)} nodes")

dataset = create_pyg_dataset(
    data_m0=lung_m0,
    data_m12=lung_m12,
    taxon_cols=taxon_cols,
    global_tree=global_tree,
    global_taxa2id=global_taxa2id,
    outcome_col=OUTCOME,
    normalization=NORMALIZATION
)

# GNN Model 
class PhyloGNN(nn.Module):
    """Phylogeny-informed Graph Neural Network (GIN)"""
    
    def __init__(self, num_embeddings, embedding_dim=24, hidden=48, 
                 covar_dim=4, dropout=0.3, n_layers=2):
        super().__init__()
        
        self.abundance_proj = nn.Sequential(nn.Linear(1, hidden), nn.ReLU())
        self.taxo_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.distance_proj = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        
        in_channels = hidden + embedding_dim + 32
        
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_channels, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
            self.convs.append(GINConv(mlp, train_eps=True))
        
        self.dropout = dropout
        
        self.head = nn.Sequential(
            nn.Linear(hidden + covar_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        abundance = x[:, 0:1]
        taxo_id = torch.clamp(x[:, 1].long(), 0, self.taxo_embedding.num_embeddings - 1)
        distance = x[:, 2:3]
        
        abundance_emb = self.abundance_proj(abundance)
        taxo_emb = self.taxo_embedding(taxo_id)
        distance_emb = self.distance_proj(distance)
        
        h = torch.cat([abundance_emb, taxo_emb, distance_emb], dim=1)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        graph_emb = global_mean_pool(h, batch)
        
        if hasattr(data, "cov") and data.cov.numel() > 0:
            cov = data.cov.to(graph_emb.device)
            if cov.dim() == 1:
                cov = cov.unsqueeze(0)
            if cov.size(0) != graph_emb.size(0):
                cov = cov.repeat(graph_emb.size(0), 1)
            combined = torch.cat([graph_emb, cov], dim=1)
        else:
            combined = graph_emb
        
        return self.head(combined)


def evaluate_loader(model, loader, device='cpu'):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            ys.append(batch.y.view(-1).cpu().numpy())
            ps.append(torch.sigmoid(out).cpu().numpy())
    
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    try:
        roc = roc_auc_score(y_true, y_pred)
        pr = average_precision_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred_binary)
    except ValueError:
        roc, pr, bacc = np.nan, np.nan, np.nan
    
    return roc, pr, bacc


def train_one_fold(model, train_loader, val_loader, optimizer, pos_weight, 
                   n_epochs, device='cpu'):
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    best_val_metric = 0
    best_state = None
    patience_counter = 0
    patience = 50
    
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            yb = batch.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(out, yb, pos_weight=pos_weight_tensor)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            val_roc, _, _ = evaluate_loader(model, val_loader, device)
            
            if not np.isnan(val_roc) and val_roc > best_val_metric:
                best_val_metric = val_roc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 10
            
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


y_all = torch.stack([d.y for d in dataset]).view(-1).numpy()
n_pos = (y_all == 1).sum()
n_neg = (y_all == 0).sum()
pos_weight = n_neg / n_pos

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_models = []
fold_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), y_all), 1):
    print("======================================================================")
    print(f"FOLD {fold_idx}/{K_FOLDS}")
    print("======================================================================")
    
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    
    model = PhyloGNN(
        num_embeddings=len(global_taxa2id),
        embedding_dim=EMBEDDING_DIM,
        hidden=HIDDEN,
        covar_dim=4,
        dropout=DROPOUT,
        n_layers=N_LAYERS
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    print("Training...")
    model = train_one_fold(model, train_loader, val_loader, optimizer, 
                          pos_weight, EPOCHS, device)
    
    train_roc, train_pr, train_bacc = evaluate_loader(model, train_loader, device)
    val_roc, val_pr, val_bacc = evaluate_loader(model, val_loader, device)
    
    print(f"\nFold {fold_idx} Results:")
    print(f"  Train: AUC={train_roc:.4f} | PR={train_pr:.4f}")
    print(f"  Val:   AUC={val_roc:.4f} | PR={val_pr:.4f}")
    
    fold_metrics.append(val_roc)
    best_models.append(copy.deepcopy(model.state_dict()))


final_model = PhyloGNN(
    num_embeddings=len(global_taxa2id),
    embedding_dim=EMBEDDING_DIM,
    hidden=HIDDEN,
    covar_dim=4,
    dropout=DROPOUT,
    n_layers=N_LAYERS
).to(device)

full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
final_roc, final_pr, final_bacc = evaluate_loader(final_model, full_loader, device)


mean_metric = np.mean(fold_metrics)
std_metric = np.std(fold_metrics)

print(f"\nCross-validation:")
print(f"  Mean Val AUC: {mean_metric:.4f} ± {std_metric:.4f}")
print(f"  Per-fold: {[f'{m:.4f}' for m in fold_metrics]}")


results_df = pd.DataFrame([{
    "method": "gnn",
    "mean_val_auc": mean_metric,
    "std_val_auc": std_metric,
    "final_auc": final_roc,
    "final_pr": final_pr,
    "fold_aucs": str(fold_metrics)
}])

results_df.to_csv(f"{OUTPUT_DIR}/gnn_{OUTCOME}.csv", index=False)
torch.save(final_model.state_dict(), f"{OUTPUT_DIR}/gnn_model.pt")

print(f"\nSaved: {OUTPUT_DIR}/gnn_{OUTCOME}.csv")
print("\nDone!")