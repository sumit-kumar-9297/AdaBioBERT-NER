#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training script for BioBERT NER with Adaptive Token-Sequence Loss
"""

import os
import json
import argparse
import logging
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

# Import from local modules
from model import CRFModel
from data_utils import get_unique_labels, read_tsv, prepare_dataset
from loss import AdaptiveWeighter, CombinedLoss
from utils import setup_directories, setup_logging, save_metrics, plot_training_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BioBERT NER with Adaptive Loss')
    
    # Data paths
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., BC4CHEMD)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory with IOB files')
    parser.add_argument('--word2vec_path', type=str, required=True, help='Path to word2vec model')
    parser.add_argument('--output_dir', type=str, default='./experiments', help='Output directory for models and results')
    
    # Model parameters
    parser.add_argument('--biobert_model', type=str, default='dmis-lab/biobert-v1.1', help='BioBERT model name from HuggingFace')
    parser.add_argument('--alpha_init', type=float, default=0.5, help='Initial value for alpha in adaptive loss')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--freeze_bert_layers', type=int, default=6, help='Number of BERT layers to freeze (0 for none)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for BERT layers')
    parser.add_argument('--alpha_lr', type=float, default=0.01, help='Learning rate for alpha parameter')
    parser.add_argument('--classifier_lr', type=float, default=1e-4, help='Learning rate for classifier layers')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=10, help='Number of steps between logging')
    parser.add_argument('--save_model_every', type=int, default=1, help='Save model every N epochs')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args, model, train_loader, dev_loader, optimizer, device, logger, experiment_dir):
    """Train the model."""
    # Set up automatic mixed precision if requested
    scaler = None
    if args.use_amp and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("Using automatic mixed precision training")
    
    # Initialize metrics tracking
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_ce_loss': [],
        'train_crf_loss': [],
        'alpha': [],
        'val_macro_f1': [],
        'val_micro_f1': []
    }
    
    best_macro_f1 = 0.0
    best_micro_f1 = 0.0
    alpha_values = []
    
    # Training loop
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
            logger=logger
        )
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dev_loader=dev_loader,
            device=device,
            args=args,
            logger=logger
        )
        
        # Track alpha value
        alpha_values.append(train_metrics['alpha'])
        
        # Update metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_metrics['loss'])
        metrics['train_ce_loss'].append(train_metrics['ce_loss'])
        metrics['train_crf_loss'].append(train_metrics['crf_loss'])
        metrics['alpha'].append(train_metrics['alpha'])
        metrics['val_macro_f1'].append(val_metrics['macro_f1'])
        metrics['val_micro_f1'].append(val_metrics['micro_f1'])
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1} results:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.6f}")
        logger.info(f"  Train CE Loss: {train_metrics['ce_loss']:.6f}")
        logger.info(f"  Train CRF Loss: {train_metrics['crf_loss']:.6f}")
        logger.info(f"  Alpha: {train_metrics['alpha']:.6f}")
        logger.info(f"  Val Macro F1: {val_metrics['macro_f1']:.6f}")
        logger.info(f"  Val Micro F1: {val_metrics['micro_f1']:.6f}")
        
        # Save metrics after each epoch
        save_metrics(metrics, os.path.join(experiment_dir, 'training_metrics.json'))
        
        # Create plot of metrics
        plot_training_metrics(metrics, os.path.join(experiment_dir, 'training_plot.png'))
        
        # Save model checkpoint
        if (epoch + 1) % args.save_model_every == 0:
            checkpoint_path = os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': train_metrics['loss'],
                'ce_loss': train_metrics['ce_loss'],
                'crf_loss': train_metrics['crf_loss'],
                'alpha': train_metrics['alpha'],
                'macro_f1': val_metrics['macro_f1'],
                'micro_f1': val_metrics['micro_f1']
            }, checkpoint_path)
            logger.info(f"Saved model checkpoint to {checkpoint_path}")
        
        # Save best model based on macro F1
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_model_path = os.path.join(experiment_dir, 'best_model_macro.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': train_metrics['loss'],
                'ce_loss': train_metrics['ce_loss'],
                'crf_loss': train_metrics['crf_loss'],
                'alpha': train_metrics['alpha'],
                'macro_f1': val_metrics['macro_f1'],
                'micro_f1': val_metrics['micro_f1']
            }, best_model_path)
            logger.info(f"New best model saved with Macro F1: {best_macro_f1:.6f}")
        
        # Also save best model based on micro F1 if different
        if val_metrics['micro_f1'] > best_micro_f1:
            best_micro_f1 = val_metrics['micro_f1']
            best_model_path = os.path.join(experiment_dir, 'best_model_micro.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': train_metrics['loss'],
                'ce_loss': train_metrics['ce_loss'],
                'crf_loss': train_metrics['crf_loss'],
                'alpha': train_metrics['alpha'],
                'macro_f1': val_metrics['macro_f1'],
                'micro_f1': val_metrics['micro_f1']
            }, best_model_path)
            logger.info(f"New best model saved with Micro F1: {best_micro_f1:.6f}")
    
    # Calculate average alpha value over all epochs
    avg_alpha = sum(alpha_values) / len(alpha_values) if alpha_values else 0
    logger.info(f"Training completed. Average alpha: {avg_alpha:.6f}, Final alpha: {alpha_values[-1]:.6f}")
    
    # Save average alpha in a separate file
    with open(os.path.join(experiment_dir, 'alpha_summary.txt'), 'w') as f:
        f.write(f"Average alpha: {avg_alpha:.6f}\n")
        f.write(f"Final alpha: {alpha_values[-1]:.6f}\n")
        f.write("\nAlpha values by epoch:\n")
        for i, alpha in enumerate(alpha_values):
            f.write(f"Epoch {i+1}: {alpha:.6f}\n")
    
    return metrics, best_macro_f1, best_micro_f1, avg_alpha, alpha_values[-1]

def train_epoch(model, train_loader, optimizer, device, epoch, args, scaler, logger):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_ce_loss = 0
    total_crf_loss = 0
    total_alpha = 0
    steps = 0
    batch_accumulation_count = 0
    
    # Use tqdm if available, otherwise use regular iteration
    try:
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    except ImportError:
        progress_bar = train_loader
        logger.info(f"Processing {len(train_loader)} batches")
    
    # Clear cache at start of epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass with AMP if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, ce_loss, crf_loss, alpha, logits = model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                # Normalize loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights after accumulating gradients
            if (batch_accumulation_count + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients 
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step optimizer and scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Clear cache periodically
                if steps % args.log_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            # Regular forward/backward without AMP
            loss, ce_loss, crf_loss, alpha, logits = model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            # Normalize loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights after accumulating gradients
            if (batch_accumulation_count + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear cache periodically
                if steps % args.log_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Update tracking variables - use original (non-normalized) loss values
        unnormalized_loss = loss * args.gradient_accumulation_steps  # De-normalize for logging
        total_loss += unnormalized_loss.item()
        total_ce_loss += ce_loss.item()
        total_crf_loss += crf_loss.item()
        total_alpha += alpha.item()
        steps += 1
        batch_accumulation_count += 1
        
        # Update progress bar if using tqdm
        if hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({
                'loss': f'{unnormalized_loss.item():.4f}', 
                'CE': f'{ce_loss.item():.4f}',
                'CRF': f'{crf_loss.item():.4f}',
                'alpha': f'{alpha.item():.4f}'
            })
        elif steps % args.log_interval == 0:
            logger.info(f"  Step {steps}/{len(train_loader)}: loss={unnormalized_loss.item():.4f}, alpha={alpha.item():.4f}")
        
        # Explicitly delete variables to free memory
        del input_ids, attention_mask, labels, loss, ce_loss, crf_loss, logits
    
    # Make sure to step optimizer for any remaining gradients
    if batch_accumulation_count % args.gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    # Calculate average metrics
    avg_loss = total_loss / steps
    avg_ce_loss = total_ce_loss / steps
    avg_crf_loss = total_crf_loss / steps
    avg_alpha = total_alpha / steps
    
    # Return metrics
    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'crf_loss': avg_crf_loss,
        'alpha': avg_alpha
    }

def evaluate(model, dev_loader, device, args, logger):
    """Evaluate the model on the development set."""
    model.eval()
    
    from sklearn.metrics import classification_report, f1_score
    import numpy as np
    
    all_true_labels = []
    all_pred_labels = []
    
    # Use tqdm if available
    try:
        from tqdm import tqdm
        progress_bar = tqdm(dev_loader, desc="Validation")
    except ImportError:
        progress_bar = dev_loader
        logger.info(f"Evaluating on {len(dev_loader)} batches")
    
    # Clear cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get predictions
            if hasattr(torch.cuda, 'amp') and args.use_amp:
                with torch.cuda.amp.autocast():
                    predictions, _ = model(input_ids, attention_mask=attention_mask)
            else:
                predictions, _ = model(input_ids, attention_mask=attention_mask)
            
            # Convert predictions to flat list
            for i, pred_seq in enumerate(predictions):
                true_seq = labels[i].cpu().numpy()
                attention = attention_mask[i].cpu().numpy()
                
                for j in range(len(true_seq)):
                    if attention[j] == 1 and true_seq[j] != -100:
                        all_true_labels.append(true_seq[j])
                        # Handle predictions
                        if j < len(pred_seq):
                            all_pred_labels.append(pred_seq[j])
                        else:
                            all_pred_labels.append(0)  # Default to O tag if prediction is missing
            
            # Free memory
            del input_ids, attention_mask, labels, predictions
            if len(all_true_labels) % (args.eval_batch_size * 10) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Get id_to_label mapping from model
    id_to_label = model.id_to_label
    
    # Calculate metrics with sklearn
    report = classification_report(
        all_true_labels,
        all_pred_labels,
        target_names=[id_to_label[i] for i in range(len(id_to_label))],
        zero_division=0,
        labels=list(range(len(id_to_label))),
        output_dict=True,
    )
    
    # Calculate macro and micro F1 scores directly
    macro_f1 = report['macro avg']['f1-score']
    micro_f1 = f1_score(all_true_labels, all_pred_labels, average='micro')
    
    # Return metrics
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'report': report
    }

def test(args, model, test_loader, device, logger, experiment_dir, label_to_id):
    """Test the model on the test set."""
    # Load best model based on macro F1
    best_model_path = os.path.join(experiment_dir, 'best_model_macro.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best macro F1 model from epoch {checkpoint['epoch']+1} with Macro F1: {checkpoint['macro_f1']:.6f}")
    else:
        logger.warning("No best macro F1 model found, using current model state")
    
    # Evaluate
    model.eval()
    
    from sklearn.metrics import classification_report, f1_score
    import numpy as np
    
    all_true_labels = []
    all_pred_labels = []
    
    # Use tqdm if available
    try:
        from tqdm import tqdm
        progress_bar = tqdm(test_loader, desc="Testing")
    except ImportError:
        progress_bar = test_loader
        logger.info(f"Testing on {len(test_loader)} batches")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get predictions
            if hasattr(torch.cuda, 'amp') and args.use_amp:
                with torch.cuda.amp.autocast():
                    predictions, _ = model(input_ids, attention_mask=attention_mask)
            else:
                predictions, _ = model(input_ids, attention_mask=attention_mask)
            
            # Convert predictions to flat list
            for i, pred_seq in enumerate(predictions):
                true_seq = labels[i].cpu().numpy()
                attention = attention_mask[i].cpu().numpy()
                
                for j in range(len(true_seq)):
                    if attention[j] == 1 and true_seq[j] != -100:
                        all_true_labels.append(true_seq[j])
                        # Handle predictions
                        if j < len(pred_seq):
                            all_pred_labels.append(pred_seq[j])
                        else:
                            all_pred_labels.append(0)
    
    # Get id_to_label mapping
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    # Calculate metrics
    report = classification_report(
        all_true_labels,
        all_pred_labels,
        target_names=[id_to_label[i] for i in range(len(id_to_label))],
        zero_division=0,
        labels=list(range(len(id_to_label))),
        output_dict=True,
    )
    
    # Calculate macro and micro F1 scores directly
    macro_f1 = report['macro avg']['f1-score']
    micro_f1 = f1_score(all_true_labels, all_pred_labels, average='micro')
    
    # Save test results
    test_results_path = os.path.join(experiment_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save as text report
    test_report_path = os.path.join(experiment_dir, 'test_report.txt')
    with open(test_report_path, 'w') as f:
        f.write(f"Test Results for {args.dataset}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Classification Report:\n")
        
        # Write report for each label
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"{label}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
            else:
                f.write(f"{label}: {metrics:.4f}\n")
        
        # Write alpha values
        if 'alpha' in checkpoint:
            f.write(f"\nFinal alpha value: {checkpoint['alpha']:.6f}\n")
        
        # Get average alpha from file
        alpha_summary_path = os.path.join(experiment_dir, 'alpha_summary.txt')
        if os.path.exists(alpha_summary_path):
            with open(alpha_summary_path, 'r') as alpha_file:
                alpha_summary = alpha_file.read()
                f.write(f"\n{alpha_summary}\n")
        
        # Write macro and micro F1
        f.write(f"\nTest Macro F1: {macro_f1:.6f}\n")
        f.write(f"Test Micro F1: {micro_f1:.6f}\n")
    
    # Log test results
    logger.info(f"Test results for {args.dataset}:")
    logger.info(f"  Macro F1: {macro_f1:.6f}")
    logger.info(f"  Micro F1: {micro_f1:.6f}")
    if 'alpha' in checkpoint:
        logger.info(f"  Final alpha value: {checkpoint['alpha']:.6f}")
    
    # Also get average alpha from file
    alpha_summary_path = os.path.join(experiment_dir, 'alpha_summary.txt')
    if os.path.exists(alpha_summary_path):
        with open(alpha_summary_path, 'r') as alpha_file:
            first_line = alpha_file.readline().strip()
            if first_line.startswith("Average alpha:"):
                avg_alpha = first_line.split(":")[-1].strip()
                logger.info(f"  Average alpha: {avg_alpha}")
    
    logger.info(f"Test results saved to {test_results_path}")
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'report': report
    }

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.dataset}_{timestamp}"
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    setup_directories(experiment_dir)
    
    # Set up logging
    logger = setup_logging(os.path.join(experiment_dir, 'training.log'))
    logger.info(f"Starting experiment {experiment_name}")
    logger.info(f"Arguments: {args}")
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load word2vec embeddings
    from gensim.models import KeyedVectors
    logger.info(f"Loading Word2Vec model from {args.word2vec_path}")
    word2vec_model = KeyedVectors.load(args.word2vec_path)
    embedding_dim = word2vec_model.vector_size
    logger.info(f"Word2Vec embedding dimension: {embedding_dim}")
    
    # Create embedding vectors tensor
    word_vectors = torch.FloatTensor(word2vec_model.wv.vectors)
    
    # Set up dataset paths
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_path = os.path.join(dataset_dir, "train.tsv")
    dev_path = os.path.join(dataset_dir, "devel.tsv")
    test_path = os.path.join(dataset_dir, "test.tsv")
    
    # Get labels
    logger.info("Getting unique labels")
    labels = get_unique_labels(train_path).union(get_unique_labels(dev_path)).union(get_unique_labels(test_path))
    label_to_id = {label: idx for idx, label in enumerate(sorted(labels))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    logger.info(f"Found {len(label_to_id)} unique labels")
    
    # Read datasets
    logger.info("Reading datasets")
    train_sentences, train_labels = read_tsv(train_path, label_to_id)
    dev_sentences, dev_labels = read_tsv(dev_path, label_to_id)
    test_sentences, test_labels = read_tsv(test_path, label_to_id)
    
    logger.info(f"Train: {len(train_sentences)} sentences")
    logger.info(f"Dev: {len(dev_sentences)} sentences")
    logger.info(f"Test: {len(test_sentences)} sentences")
    
    # Create tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.biobert_model, use_fast=True)
    
    # Create datasets
    train_dataset = prepare_dataset(train_sentences, train_labels, tokenizer, max_len=args.max_seq_length)
    dev_dataset = prepare_dataset(dev_sentences, dev_labels, tokenizer, max_len=args.max_seq_length)
    test_dataset = prepare_dataset(test_sentences, test_labels, tokenizer, max_len=args.max_seq_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    # Initialize model
    from model import CRFModel
    
    model_config = {
        "num_labels": len(label_to_id),
        "id_to_label": id_to_label,
        "word_vectors": word_vectors,
        "embedding_dim": embedding_dim,
        "biobert_model": args.biobert_model,
        "dropout_rate": args.dropout_rate,
        "freeze_bert_layers": args.freeze_bert_layers
    }
    
    logger.info("Initializing model")
    model = CRFModel(
        config=model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha_init=args.alpha_init,
        alpha_lr=args.alpha_lr,
        classifier_lr=args.classifier_lr
    )
    model.to(device)
    
    # Create optimizer
    optimizer = model.get_optimizer()
    
    # Train model
    logger.info("Starting training")
    metrics, best_macro_f1, best_micro_f1, avg_alpha, final_alpha = train(
        args, model, train_loader, dev_loader, optimizer, device, logger, experiment_dir
    )
    
    # Test model
    logger.info("Testing best model")
    test_results = test(args, model, test_loader, device, logger, experiment_dir, label_to_id)
    
    # Log final results
    logger.info("Final Results:")
    logger.info(f"  Best Validation Macro F1: {best_macro_f1:.6f}")
    logger.info(f"  Best Validation Micro F1: {best_micro_f1:.6f}")
    logger.info(f"  Test Macro F1: {test_results['macro_f1']:.6f}")
    logger.info(f"  Test Micro F1: {test_results['micro_f1']:.6f}")
    logger.info(f"  Final Alpha: {final_alpha:.6f}")
    logger.info(f"  Average Alpha: {avg_alpha:.6f}")
    
    logger.info(f"Experiment completed. Results saved to {experiment_dir}")
    
    return metrics, test_results

if __name__ == "__main__":
    main()