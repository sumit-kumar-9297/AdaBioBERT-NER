#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for BioBERT NER
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np

def setup_directories(experiment_dir):
    """
    Create experiment directories if they don't exist.
    
    Args:
        experiment_dir: Path to the experiment directory.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'results'), exist_ok=True)

def setup_logging(log_path):
    """
    Set up logging to file and console.
    
    Args:
        log_path: Path to the log file.
        
    Returns:
        logger: Configured logger.
    """
    # Create logger
    logger = logging.getLogger('biobert_ner')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_metrics(metrics, file_path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics.
        file_path: Path to the output file.
    """
    # Convert any non-serializable values to strings
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            serializable_metrics[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

def plot_training_metrics(metrics, file_path):
    """
    Create and save plots of training metrics.
    
    Args:
        metrics: Dictionary of metrics.
        file_path: Path to the output file.
    """
    # Set up figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    axes[0, 0].plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Total Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot CE and CRF losses
    axes[0, 1].plot(metrics['epoch'], metrics['train_ce_loss'], 'r-', label='CE Loss')
    axes[0, 1].plot(metrics['epoch'], metrics['train_crf_loss'], 'g-', label='CRF Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot alpha
    axes[1, 0].plot(metrics['epoch'], metrics['alpha'], 'k-')
    axes[1, 0].set_title('Alpha Value')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Alpha')
    axes[1, 0].set_ylim([0, 1])  # Alpha is bounded between 0 and 1
    axes[1, 0].grid(True)
    
    # Plot validation F1 (both macro and micro)
    if 'val_macro_f1' in metrics and 'val_micro_f1' in metrics:
        axes[1, 1].plot(metrics['epoch'], metrics['val_macro_f1'], 'm-', label='Macro F1')
        axes[1, 1].plot(metrics['epoch'], metrics['val_micro_f1'], 'c-', label='Micro F1')
        axes[1, 1].legend()
    elif 'val_f1' in metrics:
        axes[1, 1].plot(metrics['epoch'], metrics['val_f1'], 'm-', label='F1 Score')
    
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_ylim([0, 1])  # F1 is bounded between 0 and 1
    axes[1, 1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create a dedicated alpha plot
    alpha_fig, alpha_ax = plt.subplots(figsize=(10, 6))
    alpha_ax.plot(metrics['epoch'], metrics['alpha'], 'k-', linewidth=2)
    alpha_ax.set_title('Alpha Value Across Training Epochs')
    alpha_ax.set_xlabel('Epoch')
    alpha_ax.set_ylabel('Alpha')
    alpha_ax.set_ylim([0, 1])  # Alpha is bounded between 0 and 1
    alpha_ax.grid(True)
    
    # Add average alpha line
    if len(metrics['alpha']) > 0:
        avg_alpha = sum(metrics['alpha']) / len(metrics['alpha'])
        alpha_ax.axhline(y=avg_alpha, color='r', linestyle='--', label=f'Average: {avg_alpha:.4f}')
        alpha_ax.legend()
    
    # Save alpha figure
    alpha_plot_path = file_path.replace('.png', '_alpha.png')
    plt.savefig(alpha_plot_path, dpi=300, bbox_inches='tight')
    plt.close(alpha_fig)

def create_repo_structure():
    """
    Create the repository directory structure.
    """
    # Create main directories
    directories = [
        'src',
        'scripts',
        'configs',
        'experiments',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create empty __init__.py files
    init_files = [
        'src/__init__.py',
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# -*- coding: utf-8 -*-\n")
    
    print("Repository structure created successfully!")

if __name__ == "__main__":
    create_repo_structure()