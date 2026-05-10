import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader
from libraries.model import load_model as base_load_model
import libraries.dataset as cld

def load_screening_model(model_dir, device='cpu'):
    """
    Loads a trained model and its parameters from the results directory.
    """
    params_path = os.path.join(model_dir, 'model_parameters.json')
    weights_path = os.path.join(model_dir, 'model.pt')
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Model parameters not found at {params_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    with open(params_path, 'r') as f:
        model_params = json.load(f)
    
    model_type = model_params.get('model_type', 'GCNN')
    # Default to 5 features as per graph.py implementation if not specified
    n_node_features = model_params.get('n_node_features', 5)
    pdropout = model_params.get('dropout', model_params.get('pdropout', 0.0))
    
    targets = model_params.get('targets', ['E_3D'])
    n_outputs = model_params.get('n_outputs', len(targets))
    
    print(f"Instantiating {model_type} with {n_node_features} features and {n_outputs} outputs...")
    
    # Load the model using the base utility
    model = base_load_model(
        model_type=model_type,
        n_node_features=n_node_features,
        pdropout=pdropout,
        device=device,
        model_name=weights_path,
        mode='eval',
        n_outputs=n_outputs
    )
    
    return model, model_params

def load_screening_dataset(dataset_path, std_params_path, model_params):
    """
    Loads dataset.pt and applies the same transformations used during training.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not os.path.exists(std_params_path):
        raise FileNotFoundError(f"Standardized parameters not found at {std_params_path}")
        
    dataset = torch.load(dataset_path, weights_only=False)
    
    with open(std_params_path, 'r') as f:
        std_params = json.load(f)
        
    epsilon = model_params.get('epsilon', 1e-6)
    
    # 1. Apply log transformation (must be done before standardization)
    for data in tqdm(dataset, desc="Log transformation", leave=False):
        data.y = torch.log(data.y + epsilon)
        
    # 2. Apply standardization
    print("Standardizing dataset...")
    # Convert lists to numpy arrays to avoid TypeError in dataset.py arithmetic
    for key in ['edge_mean', 'feat_mean', 'target_mean', 'edge_std', 'feat_std', 'target_std']:
        if key in std_params:
            std_params[key] = np.array(std_params[key])
            
    dataset_std = cld.standardize_dataset_from_keys(dataset, std_params)
    
    # 3. Ensure float32 (to avoid "Double vs Float" error during inference)
    for data in tqdm(dataset_std, desc="Ensuring float32", leave=False):
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y = data.y.float()
    
    return dataset_std, std_params

def run_inference(model, dataset, device, std_params, model_params, batch_size=32):
    """
    Runs inference and returns a list of results with metadata and real-space predictions.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    target_mean = np.array(std_params['target_mean'])
    target_std = np.array(std_params['target_std'])
    scale = std_params.get('scale', 1.0)
    target_factor = target_std / scale
    epsilon = model_params.get('epsilon', 1e-6)
    targets_order = model_params.get('targets', ['E_3D'])
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch = batch.to(device)
            out = model(batch) # Normalized log-space
            
            # Move to CPU for processing
            out_np = out.cpu().numpy()
            
            # Rescale back to log-space: y_log = out * factor + mean
            out_log = out_np * target_factor + target_mean
            
            # Inverse log transform: y_real = exp(y_log) - epsilon
            out_real = np.exp(out_log) - epsilon
            
            # Batch metadata
            # Note: PyG Batch object has 'label' attribute which is a list of labels
            labels = batch.label
            
            for i in range(len(labels)):
                mat_info = extract_candidate_metadata(labels[i])
                res = {
                    'material': mat_info['material'],
                    'symmetry': mat_info['symmetry'],
                    'label': labels[i]
                }
                # Add individual energy predictions
                for j, target_name in enumerate(targets_order):
                    res[target_name] = out_real[i, j]
                
                results.append(res)
                
    return results

def extract_candidate_metadata(label):
    """
    Extracts material name and symmetry from the graph label.
    Expected format: "MaterialName Symmetry"
    """
    parts = label.split(' ', 1)
    if len(parts) == 2:
        return {'material': parts[0], 'symmetry': parts[1]}
    else:
        return {'material': label, 'symmetry': 'Unknown'}

def rank_candidates(results, target='E_3D', weights=None):
    """
    Ranks candidates based on the target energy or a weighted sum of energies.
    
    Args:
        results (list): List of dictionaries with predictions.
        target (str): The single target to use if weights are None.
        weights (dict, optional): A dictionary of {target_name: weight}. 
                                  If provided, ranking is based on weighted sum.
    
    Returns:
        tuple: (DataFrame of results, string describing the ranking formula)
    """
    df = pd.DataFrame(results)
    ranking_label = ""
    
    if weights:
        # Calculate weighted sum
        df['ranking_score'] = 0.0
        applied_weights_str = []
        for t, w in weights.items():
            if t in df.columns:
                df['ranking_score'] += df[t] * w
                if w != 0:
                    applied_weights_str.append(f"{w}*{t}")
        
        ranking_label = " + ".join(applied_weights_str)
        print(f"Ranking by weighted sum: {ranking_label}")
    else:
        # Single target ranking
        if target not in df.columns:
            # Fallback
            available_targets = [k for k in results[0].keys() if k.startswith('E_')]
            target = available_targets[0] if available_targets else target
            print(f"Warning: Target {target} not found, using {target} for ranking.")
        
        df['ranking_score'] = df[target]
        ranking_label = target
        print(f"Ranking by single target: {target}")

    # Rank by score (ascending: lower energy is usually better for activation)
    df = df.sort_values(by='ranking_score').reset_index(drop=True)
    df['rank'] = df.index + 1
    
    return df, ranking_label

def write_candidates_txt(df, top_n, output_path):
    """
    Writes the top N candidates to candidates.txt in the required format.
    Format: "Formula Symmetry"
    """
    top_df = df.head(top_n)
    with open(output_path, 'w') as f:
        for _, row in top_df.iterrows():
            f.write(f"{row['material']} {row['symmetry']}\n")
    print(f"Exported top {top_n} candidates to {output_path}")

def write_predictions_csv(df, output_path):
    """
    Writes all predictions to candidate_predictions.csv.
    """
    # Ensure columns order
    energy_cols = [c for c in df.columns if c.startswith('E_')]
    final_cols = ['material', 'symmetry'] + sorted(energy_cols) + ['ranking_score', 'rank']
    
    # Filter columns to only what exists
    final_cols = [c for c in final_cols if c in df.columns]
    
    df[final_cols].to_csv(output_path, index=False)
    print(f"Exported all predictions to {output_path}")

def plot_energy_distributions(results):
    """
    Plots the distribution of predicted energies for each target.
    Clips the X-axis to the 99th percentile for better visibility.
    """
    df = pd.DataFrame(results)
    energy_cols = [c for c in df.columns if c.startswith('E_')]
    
    # Calculate global 99th percentile to clip outlier view
    all_values = df[energy_cols].values.flatten()
    upper_limit = np.percentile(all_values, 99)
    
    plt.figure(figsize=(10, 5))
    for col in energy_cols:
        sns.histplot(df[col], kde=True, label=col, alpha=0.4)
    
    plt.title("Distribution of Predicted Activation Energies")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Frequency")
    plt.xlim(0, upper_limit) # Shorten the X-axis
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_top_candidates(df, top_n=10, label="Ranking Score"):
    """
    Plots a bar chart of the top N candidates and their ranking scores.
    """
    top_df = df.head(top_n).copy()
    top_df['display_label'] = top_df['material'] + " (" + top_df['symmetry'] + ")"
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_df, x='ranking_score', y='display_label', palette='viridis')
    
    plt.title(f"Top {top_n} Materials by {label}")
    plt.xlabel(label)
    plt.ylabel("Material")
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
