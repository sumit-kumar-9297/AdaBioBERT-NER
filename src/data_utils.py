#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing utilities for BioBERT NER
"""

import torch
from torch.utils.data import Dataset

def get_unique_labels(file_path):
    """
    Get the set of unique labels from a TSV file.
    
    Args:
        file_path: Path to the TSV file with token-label pairs.
        
    Returns:
        set: Set of unique labels.
    """
    unique_labels = set()
    with open(file_path, "r") as file:
        for line in file:
            if line.strip():
                try:
                    label = line.strip().split("\t")[-1]
                    unique_labels.add(label)
                except IndexError:
                    pass  # Skip malformed lines
    return unique_labels

def read_tsv(file_path, label_to_id):
    """
    Read sentences and labels from a TSV file.
    
    Args:
        file_path: Path to the TSV file with token-label pairs.
        label_to_id: Dictionary mapping labels to IDs.
        
    Returns:
        tuple: List of sentences (token lists) and list of label ID lists.
    """
    sentences, tags = [], []
    with open(file_path, "r") as f:
        tokens, labels = [], []
        for line in f:
            if line.strip() == "":
                if tokens:
                    sentences.append(tokens)
                    tags.append([label_to_id[label] for label in labels])
                    tokens, labels = [], []
            else:
                try:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        token, label = parts[0], parts[-1]
                        tokens.append(token)
                        labels.append(label)
                    else:
                        print(f"Skipping malformed line: {line.strip()}")
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
        if tokens:
            sentences.append(tokens)
            tags.append([label_to_id[label] for label in labels])
    return sentences, tags

class prepare_dataset(Dataset):
    """
    Dataset for tokenized sentences and labels.
    
    Handles the conversion of tokens to input IDs and aligning labels with subword tokens.
    """
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.labels[idx]

        # Tokenize the input
        tokenized = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            is_split_into_words=True,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Align labels with tokens
        label_ids = [-100] * len(input_ids)
        word_ids = tokenized.word_ids(batch_index=0)
        
        # Assign a label to each token
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(labels):
                # Only assign the label to the first subword of each word
                if i == 0 or word_ids[i-1] != word_idx:
                    label_ids[i] = labels[word_idx]
                else:
                    # For other subwords, use the same label if needed
                    # or use -100 to ignore them in loss computation
                    label_ids[i] = labels[word_idx]  # or use -100 here

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }