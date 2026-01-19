#gnn_utils.py

"""
GNN Utility Functions
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from ete3 import Tree


def normalize_column_names(df):
    """Normalize column names (++ fix Excel transformations)"""
    df.columns = df.columns.str.replace("..", ": ", regex=False)
    return df


def normalize_taxon_name(name):
    """Normalize taxon names"""
    return (name
            .replace("..", ":")
            .replace(".", "-")
            .replace(",", "")
            .replace("\xa0", "")
            .strip())


def build_global_taxonomic_tree(hierarchy_df):
    """
    Build global taxonomic tree from hierarchy table
    
    Args:
        hierarchy_df: df with taxonomy ranks (ex: Kingdom) + mapping 
    
    Returns:
        tree: ete3.Tree object
        taxa2id: dict mapping taxon name to global ID
        id2taxa: dict mapping global ID to taxon name
    """
    tree = Tree(name="Root")
    taxa2id = {"Root": 0}
    id2taxa = {0: "Root"}
    
    ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
    
    for _, row in hierarchy_df.iterrows():
        path = [row[r] for r in ranks if pd.notna(row[r])]
        
        current = tree
        for taxon in path:
            taxon_str = str(taxon)
            matching = [c for c in current.children if c.name == taxon_str]
            if not matching:
                current = current.add_child(name=taxon_str)
            else:
                current = matching[0]
            
            if taxon_str not in taxa2id:
                taxa2id[taxon_str] = len(taxa2id)
                id2taxa[len(id2taxa)] = taxon_str
    
    return tree, taxa2id, id2taxa


def normalize_abundance(abundance_data, method="zscore"):
    """
    Normalize abundance values
    
    Args:
        abundance_data: DataFrame with bacterial abundance columns (b_*)
        method: "none", "log1p", or "zscore"
    
    Returns:
        Normalized DataFrame
    """
    taxon_cols = [c for c in abundance_data.columns if c.startswith("b_")]
    normalized = abundance_data.copy()
    
    if method == "log1p":
        for col in taxon_cols:
            normalized[col] = np.log1p(normalized[col].astype(float))
    elif method == "zscore":
        for col in taxon_cols:
            vals = normalized[col].astype(float).values
            mean, std = np.nanmean(vals), np.nanstd(vals) + 1e-8
            normalized[col] = (vals - mean) / std
    
    return normalized


def build_patient_graph(patient_row, taxon_cols, global_tree, global_taxa2id, threshold=1e-6):
    """
    Build patient-specific phylogenetic graph containing only present taxa
    
    Args:
        patient_row: Series with patient's abundance values
        taxon_cols: List of bacterial column names
        global_tree: Global taxonomic tree
        global_taxa2id: Global taxon ID mapping
        threshold: Minimum abundance to consider taxon present
    
    Returns:
        node_features: Tensor [N, 3] (abundance, taxo_id, distance)
        edge_index: Tensor [2, E] (graph edges)
        n_nodes: Number of nodes
    """

    present_taxa = []
    for col in taxon_cols:
        if patient_row[col] > threshold:
            taxon_name = col.replace("b_", "")
            present_taxa.append(taxon_name)
    
    if len(present_taxa) == 0:
        # Empty graph (only root node)
        node_features = np.array([[0.0, 0, 0.0]])
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(node_features, dtype=torch.float), edge_index, 1
    

    subtree = global_tree.copy()
    all_leaves = [n.name for n in subtree.get_leaves()]
    leaves_to_keep = [n for n in all_leaves if n in present_taxa]
    
    if len(leaves_to_keep) == 0:
        node_features = np.array([[0.0, 0, 0.0]])
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(node_features, dtype=torch.float), edge_index, 1
    
    subtree.prune(leaves_to_keep, preserve_branch_length=True)
    

    local_taxa2id = {}
    for i, node in enumerate(subtree.traverse()):
        local_taxa2id[node.name] = i
    
    # Build node features and edges
    node_features = []
    edges = []
    
    for node in subtree.traverse():
        local_id = local_taxa2id[node.name]
        global_id = global_taxa2id.get(node.name, 0)
        
        if node.is_leaf():
            abundance = patient_row.get(f"b_{node.name}", 0.0)
        else:
            abundance = sum([patient_row.get(f"b_{leaf.name}", 0.0) 
                           for leaf in node.get_leaves()])
        
        taxo_id = global_id
        distance = float(node.get_distance(subtree))
        node_features.append([abundance, taxo_id, distance])
        
        if not node.is_root():
            parent_id = local_taxa2id[node.up.name]
            edges.append([parent_id, local_id])
    

    node_features = np.array(node_features, dtype=float)
    if node_features[:, 0].max() > 0:
        node_features[:, 0] /= node_features[:, 0].max()
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return torch.tensor(node_features, dtype=torch.float), edge_index, len(local_taxa2id)


def prepare_covariates(data_m0):
    """Prepare clinical covariates for GNN"""
    age_num = pd.to_numeric(data_m0["Age"], errors="coerce")
    
    covars = pd.DataFrame({
        "id_patient": data_m0["id_patient"],
        "Age": age_num,
        "Age2": age_num ** 2,
        "SexM": (data_m0["Sex"].astype(str).str.upper().str.startswith("M")).astype(int),
        "bmi_zscore": pd.to_numeric(data_m0.get("bmi_zscore", np.nan), errors="coerce")
       # "Shannon_16S": pd.to_numeric(data_m0.get("Shannon_16S", np.nan), errors="coerce"),
        # "Pielou_16S": pd.to_numeric(data_m0.get("Pielou_16S", np.nan), errors="coerce"),
        # "Simpson_16S": pd.to_numeric(data_m0.get("Simpson_16S", np.nan), errors="coerce"),
        # "Chao1_16S": pd.to_numeric(data_m0.get("Chao1_16S", np.nan), errors="coerce"),
    })
    
    for col in ["Age", "Age2", "SexM", "bmi_zscore"]:#, #"Shannon_16S", "Pielou_16S", "Simpson_16S", "Chao1_16S"]:
        covars[col] = covars[col].fillna(0.0).astype(float)
    
    return covars.reset_index(drop=True)


def create_pyg_dataset(data_m0, data_m12, taxon_cols, global_tree, global_taxa2id, 
                       outcome_col, normalization="zscore"):
    """
    Create PyTorch Geometric dataset from patient data
    
    Args:
        data_m0: Baseline data
        data_m12: Follow-up data  
        taxon_cols: List of bacterial column names
        global_tree: Global taxonomic tree
        global_taxa2id: Global taxon ID mapping
        outcome_col: Name of outcome column
        normalization: Abundance normalization method
    
    Returns:
        List of PyG Data objects
    """

    normalized_data = normalize_abundance(data_m0[["id_patient"] + taxon_cols], method=normalization)
    covars = prepare_covariates(data_m0)
    y_m12 = data_m12[["id_patient", outcome_col]].rename(columns={outcome_col: "y_target"})
    
    merged = normalized_data.merge(y_m12, on="id_patient", how="inner")
    merged = merged.merge(covars, on="id_patient", how="inner").reset_index(drop=True)
    
    
    dataset = []
    for i, (idx, row) in enumerate(merged.iterrows()):
        
        node_features, edge_index, n_nodes = build_patient_graph(
            row, taxon_cols, global_tree, global_taxa2id
        )
        cov_values = row[["Age", "Age2", "SexM", "bmi_zscore"]].values.astype(float)
        cov = torch.tensor(cov_values, dtype=torch.float)
        
        y_val = torch.tensor([float(row["y_target"])], dtype=torch.float)
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=y_val,
            cov=cov.unsqueeze(0),
            id_patient=row["id_patient"],
            num_embeddings=len(global_taxa2id),
        )
        
        dataset.append(data)
    
    print(f"\nDataset created.")
    
    return dataset