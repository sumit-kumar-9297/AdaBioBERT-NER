#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced data processing utilities for BioBERT NER with proper Word2Vec integration
"""

import torch
from torch.utils.data import Dataset
import logging

def get_unique_labels(file_path):
    """
    Get the set of unique labels from a TSV file.
    
    Args:
        file_path: Path to the TSV file with token-label pairs.
        
    Returns:
        set: Set of unique labels.
    """
    unique_labels = set()
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line.strip():
                    try:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            label = parts[-1]
                            unique_labels.add(label)
                    except IndexError:
                        print(f"Warning: Skipping malformed line {line_num}: {line.strip()}")
                        continue
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise
    
    return unique_labels

def read_tsv(file_path, label_to_id):
    """
    Read sentences and labels from a TSV file with enhanced error handling.
    
    Args:
        file_path: Path to the TSV file with token-label pairs.
        label_to_id: Dictionary mapping labels to IDs.
        
    Returns:
        tuple: List of sentences (token lists) and list of label ID lists.
    """
    sentences, tags = [], []
    
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            tokens, labels = [], []
            line_num = 0
            
            for line in f:
                line_num += 1
                line = line.strip()
                
                if line == "":
                    # End of sentence
                    if tokens:
                        if len(tokens) != len(labels):
                            print(f"Warning: Token-label mismatch at line {line_num}. "
                                  f"Tokens: {len(tokens)}, Labels: {len(labels)}")
                        else:
                            sentences.append(tokens)
                            try:
                                tags.append([label_to_id[label] for label in labels])
                            except KeyError as e:
                                print(f"Warning: Unknown label '{e}' at line {line_num}. Skipping sentence.")
                                continue
                        tokens, labels = [], []
                else:
                    try:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            token, label = parts[0], parts[-1]
                            
                            # Basic validation
                            if token.strip() and label.strip():
                                tokens.append(token.strip())
                                labels.append(label.strip())
                            else:
                                print(f"Warning: Empty token or label at line {line_num}: {line}")
                        else:
                            print(f"Warning: Insufficient columns at line {line_num}: {line}")
                    except Exception as e:
                        print(f"Warning: Error parsing line {line_num}: {line}. Error: {e}")
                        continue
            
            # Handle last sentence if file doesn't end with empty line
            if tokens:
                if len(tokens) == len(labels):
                    sentences.append(tokens)
                    try:
                        tags.append([label_to_id[label] for label in labels])
                    except KeyError as e:
                        print(f"Warning: Unknown label '{e}' in final sentence. Skipping.")
                else:
                    print(f"Warning: Token-label mismatch in final sentence. "
                          f"Tokens: {len(tokens)}, Labels: {len(labels)}")
                          
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise
    
    print(f"Successfully read {len(sentences)} sentences from {file_path}")
    return sentences, tags

def validate_dataset(sentences, labels, dataset_name="dataset"):
    """
    Validate dataset consistency.
    
    Args:
        sentences: List of token lists
        labels: List of label ID lists
        dataset_name: Name of dataset for logging
        
    Returns:
        bool: True if dataset is valid
    """
    if len(sentences) != len(labels):
        print(f"Error: Sentence-label count mismatch in {dataset_name}. "
              f"Sentences: {len(sentences)}, Labels: {len(labels)}")
        return False
    
    invalid_count = 0
    for i, (sent, labs) in enumerate(zip(sentences, labels)):
        if len(sent) != len(labs):
            print(f"Warning: Length mismatch in {dataset_name} sentence {i}. "
                  f"Tokens: {len(sent)}, Labels: {len(labs)}")
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"Found {invalid_count} sentences with length mismatches in {dataset_name}")
        return False
    
    print(f"Dataset {dataset_name} validation passed")
    return True

class prepare_dataset(Dataset):
    """
    Enhanced dataset for tokenized sentences and labels with better subword handling.
    
    Handles the conversion of tokens to input IDs and aligning labels with subword tokens.
    """
    def __init__(self, sentences, labels, tokenizer, max_len=128, 
                 label_all_tokens=True, verbose=False):
        """
        Initialize dataset.
        
        Args:
            sentences: List of token lists
            labels: List of label ID lists  
            tokenizer: Tokenizer to use
            max_len: Maximum sequence length
            label_all_tokens: Whether to label all subword tokens or just the first
            verbose: Whether to print verbose information
        """
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_all_tokens = label_all_tokens
        self.verbose = verbose
        
        # Validate inputs
        assert len(sentences) == len(labels), "Sentences and labels must have same length"
        
        if verbose:
            print(f"Dataset initialized with {len(sentences)} samples")
            print(f"Max length: {max_len}")
            print(f"Label all tokens: {label_all_tokens}")
            
            # Sample statistics
            sent_lengths = [len(sent) for sent in sentences]
            print(f"Sentence length stats: min={min(sent_lengths)}, "
                  f"max={max(sent_lengths)}, avg={sum(sent_lengths)/len(sent_lengths):.1f}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.labels[idx]
        
        # Ensure consistent lengths
        assert len(tokens) == len(labels), f"Token-label mismatch at index {idx}"

        # Tokenize with alignment tracking
        try:
            tokenized = self.tokenizer(
                tokens,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                is_split_into_words=True,
                return_tensors="pt"
            )
        except Exception as e:
            if self.verbose:
                print(f"Tokenization error at index {idx}: {e}")
                print(f"Tokens: {tokens}")
            raise

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Align labels with tokenized input
        label_ids = self._align_labels_with_tokens(
            tokenized, labels, idx
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
    
    def _align_labels_with_tokens(self, tokenized, labels, idx):
        """
        Align labels with tokenized input, handling subword tokens properly.
        
        Args:
            tokenized: Tokenizer output
            labels: Original labels
            idx: Sample index for debugging
            
        Returns:
            list: Aligned label IDs
        """
        # Initialize with -100 (ignore index)
        label_ids = [-100] * len(tokenized["input_ids"][0])
        
        # Get word IDs for alignment
        word_ids = tokenized.word_ids(batch_index=0)
        
        if word_ids is None:
            if self.verbose:
                print(f"Warning: No word IDs available for sample {idx}")
            return label_ids
        
        previous_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                continue
            elif word_idx >= len(labels):
                # Word index exceeds label length (due to truncation)
                if self.verbose and word_idx == len(labels):
                    print(f"Warning: Truncation occurred at sample {idx}")
                continue
            elif word_idx != previous_word_idx:
                # First subword token of a word
                label_ids[i] = labels[word_idx]
            elif self.label_all_tokens:
                # Subsequent subword tokens of the same word
                label_ids[i] = labels[word_idx]
            # else: keep -100 for subsequent subword tokens
            
            previous_word_idx = word_idx
        
        return label_ids

def create_dataset_splits(sentences, labels, train_ratio=0.8, dev_ratio=0.1, 
                         test_ratio=0.1, random_seed=42):
    """
    Create train/dev/test splits from sentences and labels.
    
    Args:
        sentences: List of token lists
        labels: List of label ID lists
        train_ratio: Proportion for training
        dev_ratio: Proportion for development
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_data, dev_data, test_data) where each is (sentences, labels)
    """
    import random
    
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set seed for reproducibility
    random.seed(random_seed)
    
    # Create indices and shuffle
    indices = list(range(len(sentences)))
    random.shuffle(indices)
    
    # Calculate split points
    n_total = len(sentences)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    dev_indices = indices[n_train:n_train + n_dev]
    test_indices = indices[n_train + n_dev:]
    
    # Create splits
    train_sentences = [sentences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    dev_sentences = [sentences[i] for i in dev_indices]
    dev_labels = [labels[i] for i in dev_indices]
    
    test_sentences = [sentences[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"Dataset split: Train={len(train_sentences)}, "
          f"Dev={len(dev_sentences)}, Test={len(test_sentences)}")
    
    return (train_sentences, train_labels), (dev_sentences, dev_labels), (test_sentences, test_labels)

def analyze_dataset(sentences, labels, label_to_id, dataset_name="dataset"):
    """
    Analyze dataset statistics.
    
    Args:
        sentences: List of token lists
        labels: List of label ID lists
        label_to_id: Label to ID mapping
        dataset_name: Name for logging
    """
    print(f"\n=== Dataset Analysis: {dataset_name} ===")
    
    # Basic statistics
    n_sentences = len(sentences)
    n_tokens = sum(len(sent) for sent in sentences)
    avg_length = n_tokens / n_sentences if n_sentences > 0 else 0
    
    print(f"Sentences: {n_sentences}")
    print(f"Total tokens: {n_tokens}")
    print(f"Average sentence length: {avg_length:.1f}")
    
    # Length distribution
    lengths = [len(sent) for sent in sentences]
    print(f"Length distribution:")
    print(f"  Min: {min(lengths) if lengths else 0}")
    print(f"  Max: {max(lengths) if lengths else 0}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2] if lengths else 0}")
    
    # Label distribution
    label_counts = {label: 0 for label in label_to_id.keys()}
    for label_seq in labels:
        for label_id in label_seq:
            # Convert back to label name
            for label_name, lid in label_to_id.items():
                if lid == label_id:
                    label_counts[label_name] += 1
                    break
    
    print(f"Label distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / n_tokens * 100) if n_tokens > 0 else 0
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print("=" * 40)

def check_tokenizer_compatibility(sentences, tokenizer, max_length=128, 
                                 sample_size=100, verbose=True):
    """
    Check tokenizer compatibility and estimate truncation effects.
    
    Args:
        sentences: List of token lists
        tokenizer: Tokenizer to check
        max_length: Maximum sequence length
        sample_size: Number of samples to check
        verbose: Whether to print detailed information
        
    Returns:
        dict: Statistics about tokenization
    """
    import random
    
    if verbose:
        print(f"\n=== Tokenizer Compatibility Check ===")
    
    # Sample sentences for analysis
    sample_sentences = random.sample(sentences, min(sample_size, len(sentences)))
    
    truncated_count = 0
    original_lengths = []
    tokenized_lengths = []
    subword_ratios = []
    
    for sent in sample_sentences:
        original_length = len(sent)
        original_lengths.append(original_length)
        
        # Tokenize
        tokenized = tokenizer(
            sent,
            truncation=False,
            padding=False,
            is_split_into_words=True,
            return_tensors="pt"
        )
        
        tokenized_length = len(tokenized["input_ids"][0])
        tokenized_lengths.append(tokenized_length)
        
        # Calculate subword ratio
        subword_ratio = tokenized_length / original_length if original_length > 0 else 1.0
        subword_ratios.append(subword_ratio)
        
        # Check if would be truncated
        if tokenized_length > max_length:
            truncated_count += 1
    
    # Calculate statistics
    stats = {
        'samples_checked': len(sample_sentences),
        'truncated_count': truncated_count,
        'truncation_rate': truncated_count / len(sample_sentences),
        'avg_original_length': sum(original_lengths) / len(original_lengths),
        'avg_tokenized_length': sum(tokenized_lengths) / len(tokenized_lengths),
        'avg_subword_ratio': sum(subword_ratios) / len(subword_ratios),
        'max_tokenized_length': max(tokenized_lengths),
    }
    
    if verbose:
        print(f"Samples checked: {stats['samples_checked']}")
        print(f"Truncation rate: {stats['truncation_rate']:.1%}")
        print(f"Average original length: {stats['avg_original_length']:.1f}")
        print(f"Average tokenized length: {stats['avg_tokenized_length']:.1f}")
        print(f"Average subword ratio: {stats['avg_subword_ratio']:.2f}")
        print(f"Max tokenized length: {stats['max_tokenized_length']}")
        
        if stats['truncation_rate'] > 0.1:
            print(f"Warning: {stats['truncation_rate']:.1%} of sentences will be truncated")
        
        print("=" * 40)
    
    return stats
