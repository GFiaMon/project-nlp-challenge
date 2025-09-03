"""
Experiment tracking utilities for consistent result logging
"""

import pandas as pd
import numpy as np
import datetime
import os
from typing import Dict, Any

def create_experiment_id(experiment_type: str, model_name: str) -> str:
    """Create a unique experiment ID"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = ''.join([word[0] for word in model_name.split()]).lower()
    return f"{experiment_type[:2]}_{model_short}_{timestamp}"

def log_experiment(experiment_data: Dict[str, Any], results_dir: str = "../models/experiment_results"):
    """
    Log an experiment with consistent format
    
    Parameters:
    experiment_data: Dictionary with experiment results
    results_dir: Directory to save results
    """
    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Create experiment ID if not provided
    if 'experiment_id' not in experiment_data:
        experiment_data['experiment_id'] = create_experiment_id(
            experiment_data['experiment_type'], 
            experiment_data['model_name']
        )
    
    # Add timestamp if not provided
    if 'timestamp' not in experiment_data:
        experiment_data['timestamp'] = datetime.datetime.now().isoformat()
    
    # Convert to DataFrame
    exp_df = pd.DataFrame([experiment_data])
    
    # Determine subdirectory based on experiment type
    exp_type = experiment_data['experiment_type']
    exp_subdir = os.path.join(results_dir, exp_type)
    os.makedirs(exp_subdir, exist_ok=True)
    
    # Save individual experiment
    exp_file = os.path.join(exp_subdir, f"{experiment_data['experiment_id']}.csv")
    exp_df.to_csv(exp_file, index=False)
    
    # Update master results file
    master_file = os.path.join(results_dir, "master_results.csv")
    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
        master_df = pd.concat([master_df, exp_df], ignore_index=True)
    else:
        master_df = exp_df
    
    master_df.to_csv(master_file, index=False)
    
    print(f"✅ Experiment logged: {experiment_data['experiment_id']}")
    return experiment_data['experiment_id']

def load_experiment_results(results_dir: str = "../models/experiment_results"):
    """Load all experiment results into a single DataFrame"""
    master_file = os.path.join(results_dir, "master_results.csv")
    
    if os.path.exists(master_file):
        return pd.read_csv(master_file)
    else:
        # If master file doesn't exist, try to create it from individual files
        all_results = []
        for exp_type in ['sentence_transformers', 'traditional_nlp']:
            exp_dir = os.path.join(results_dir, exp_type)
            if os.path.exists(exp_dir):
                for file in os.listdir(exp_dir):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(exp_dir, file))
                        all_results.append(df)
        
        if all_results:
            master_df = pd.concat(all_results, ignore_index=True)
            master_df.to_csv(master_file, index=False)
            return master_df
        else:
            return pd.DataFrame()

def get_best_experiment(results_df: pd.DataFrame, primary_metric: str = 'f1_score'):
    """Get the best experiment based on the primary metric"""
    if results_df.empty:
        return None
    
    return results_df.loc[results_df[primary_metric].idxmax()]