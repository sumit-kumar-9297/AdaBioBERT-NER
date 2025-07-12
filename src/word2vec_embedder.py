#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuned Word2Vec embedder that always fine-tunes on training data
Simple and focused approach: fine-tune -> use embeddings -> zeros for remaining OOV
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List
import logging
from gensim.models import Word2Vec, KeyedVectors
import os
import hashlib

logger = logging.getLogger(__name__)

def extract_keyed_vectors(word2vec_model):
    """Extract KeyedVectors from Word2Vec model, handling both cases."""
    if hasattr(word2vec_model, 'wv'):
        return word2vec_model.wv
    else:
        return word2vec_model

class FineTunedWord2VecEmbedder(nn.Module):
    """
    Word2Vec embedder that fine-tunes on training data and uses zeros for remaining OOV.
    
    Simple approach:
    1. Load pre-trained Word2Vec
    2. Fine-tune on training data (adds vocabulary + improves embeddings)
    3. Create embedding matrix with fine-tuned vectors
    4. Use zeros for any remaining OOV tokens
    """
    
    def __init__(self, pretrained_word2vec_path, training_sentences, tokenizer, 
                 finetune_epochs=5, min_count=1, max_vocab_size=50000, cache_dir="./word2vec_cache"):
        """
        Initialize with fine-tuning on training data.
        
        Args:
            pretrained_word2vec_path: Path to pre-trained Word2Vec model
            training_sentences: List of training sentences (list of token lists)
            tokenizer: BioBERT tokenizer
            finetune_epochs: Number of epochs to fine-tune Word2Vec
            min_count: Minimum count for new vocabulary words
            max_vocab_size: Maximum vocabulary size for embedding matrix
            cache_dir: Directory to cache fine-tuned models
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info("=== Fine-tuned Word2Vec Embedder ===")
        
        # Create cache key based on training data and parameters
        cache_key = self._create_cache_key(training_sentences, finetune_epochs, min_count)
        cached_model_path = os.path.join(cache_dir, f"finetuned_w2v_{cache_key}.model")
        
        # Try to load cached fine-tuned model
        if os.path.exists(cached_model_path):
            logger.info(f"Loading cached fine-tuned model from {cached_model_path}")
            try:
                self.word2vec_model = Word2Vec.load(cached_model_path)
                logger.info("✓ Cached model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")
                logger.info("Will fine-tune from scratch...")
                self.word2vec_model = self._finetune_model(pretrained_word2vec_path, training_sentences, 
                                                         finetune_epochs, min_count, cached_model_path)
        else:
            logger.info("No cached model found, fine-tuning from scratch...")
            self.word2vec_model = self._finetune_model(pretrained_word2vec_path, training_sentences, 
                                                     finetune_epochs, min_count, cached_model_path)
        
        # Create embedding matrix
        self.embedding_dim = self.word2vec_model.wv.vector_size
        self._create_embedding_matrix(tokenizer, max_vocab_size)
        
        logger.info("=== Fine-tuned Word2Vec Embedder Ready ===")
    
    def _create_cache_key(self, training_sentences, finetune_epochs, min_count):
        """Create a cache key based on training data and parameters."""
        # Use first 100 sentences for hash to avoid memory issues
        sample_sentences = str(training_sentences[:100])
        params_str = f"epochs_{finetune_epochs}_mincount_{min_count}"
        combined_str = sample_sentences + params_str
        return hashlib.md5(combined_str.encode()).hexdigest()[:12]
    
    def _finetune_model(self, pretrained_path, training_sentences, epochs, min_count, cache_path):
        """Fine-tune Word2Vec model on training data."""
        
        # Step 1: Load pre-trained model
        logger.info(f"Loading pre-trained Word2Vec from {pretrained_path}")
        try:
            if pretrained_path.endswith('.model'):
                # Full Word2Vec model
                word2vec_model = Word2Vec.load(pretrained_path)
            else:
                # KeyedVectors - convert to trainable Word2Vec
                keyed_vectors = KeyedVectors.load(pretrained_path)
                logger.info("Converting KeyedVectors to trainable Word2Vec model...")
                word2vec_model = self._create_trainable_from_vectors(keyed_vectors)
        except Exception as e:
            # Try alternative loading methods
            try:
                full_model = Word2Vec.load(pretrained_path)
                word2vec_model = full_model
            except:
                try:
                    keyed_vectors = KeyedVectors.load(pretrained_path)
                    word2vec_model = self._create_trainable_from_vectors(keyed_vectors)
                except:
                    raise ValueError(f"Could not load Word2Vec model from {pretrained_path}: {e}")
        
        original_vocab_size = len(word2vec_model.wv.key_to_index)
        logger.info(f"Original vocabulary size: {original_vocab_size:,}")
        
        # Step 2: Prepare training sentences
        logger.info("Preparing training sentences for Word2Vec...")
        word2vec_sentences = self._prepare_training_sentences(training_sentences)
        logger.info(f"Prepared {len(word2vec_sentences):,} sentences for fine-tuning")
        
        # Step 3: Fine-tune the model
        logger.info(f"Fine-tuning Word2Vec for {epochs} epochs...")
        
        # Update vocabulary with new words from training data
        word2vec_model.build_vocab(word2vec_sentences, update=True, min_count=min_count)
        
        # Fine-tune the model
        word2vec_model.train(word2vec_sentences, total_examples=len(word2vec_sentences), epochs=epochs)
        
        new_vocab_size = len(word2vec_model.wv.key_to_index)
        added_words = new_vocab_size - original_vocab_size
        
        logger.info(f"Fine-tuning complete!")
        logger.info(f"  Added {added_words:,} new words")
        logger.info(f"  New vocabulary size: {new_vocab_size:,}")
        
        # Step 4: Cache the fine-tuned model
        try:
            word2vec_model.save(cache_path)
            logger.info(f"Fine-tuned model cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Could not cache model: {e}")
        
        return word2vec_model
    
    def _create_trainable_from_vectors(self, keyed_vectors):
        """Create a trainable Word2Vec model from KeyedVectors."""
        model = Word2Vec(
            vector_size=keyed_vectors.vector_size,
            window=5,
            min_count=1,
            workers=4,
            sg=1,  # Skip-gram
            epochs=0
        )
        
        # Build vocabulary and copy vectors
        vocab_list = list(keyed_vectors.key_to_index.keys())
        model.build_vocab([vocab_list], update=False)
        
        # Copy pre-trained vectors
        model.wv.vectors = keyed_vectors.vectors.copy()
        model.wv.key_to_index = keyed_vectors.key_to_index.copy()
        model.wv.index_to_key = keyed_vectors.index_to_key.copy()
        
        return model
    
    def _prepare_training_sentences(self, training_sentences):
        """Prepare sentences for Word2Vec training."""
        word2vec_sentences = []
        
        for sentence in training_sentences:
            # Clean and prepare tokens
            cleaned_tokens = []
            for token in sentence:
                if len(token) > 1:  # Skip very short tokens
                    # Add main token
                    cleaned_tokens.append(token.lower())
                    
                    # Handle compound words
                    for separator in ['-', '_', '/']:
                        if separator in token:
                            parts = token.split(separator)
                            for part in parts:
                                if len(part) > 1:
                                    cleaned_tokens.append(part.lower())
            
            # Only use sentences with sufficient tokens
            if len(cleaned_tokens) > 2:
                word2vec_sentences.append(cleaned_tokens)
        
        return word2vec_sentences
    
    def _create_embedding_matrix(self, tokenizer, max_vocab_size):
        """Create embedding matrix from fine-tuned Word2Vec."""
        logger.info("Creating embedding matrix from fine-tuned Word2Vec...")
        
        # Get BioBERT vocabulary
        vocab = tokenizer.get_vocab()
        vocab_size = min(len(vocab), max_vocab_size)
        
        # Initialize embedding matrix with zeros
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
        
        # Get fine-tuned Word2Vec vocabulary
        word2vec_vocab = set(self.word2vec_model.wv.key_to_index.keys())
        
        # Statistics
        exact_hits = 0
        compound_hits = 0
        oov_count = 0
        
        for token, token_id in vocab.items():
            if token_id >= vocab_size:
                continue
                
            # Handle special tokens with zeros
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                # Keep as zeros (already initialized)
                continue
            
            # Clean token (remove ## prefix for subwords)
            clean_token = token[2:] if token.startswith('##') else token
            
            # Try exact matches (original and lowercase)
            found = False
            for candidate in [clean_token, clean_token.lower()]:
                if candidate in word2vec_vocab:
                    embedding_matrix[token_id] = self.word2vec_model.wv[candidate]
                    exact_hits += 1
                    found = True
                    break
            
            if not found:
                # Try compound word decomposition
                compound_embedding = self._get_compound_embedding(clean_token, word2vec_vocab)
                if compound_embedding is not None:
                    embedding_matrix[token_id] = compound_embedding
                    compound_hits += 1
                    found = True
            
            if not found:
                # Remains as zero vector (already initialized)
                oov_count += 1
        
        # Create embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), 
            freeze=False  # Allow further fine-tuning during training
        )
        
        total_coverage = exact_hits + compound_hits
        logger.info(f"Embedding matrix created:")
        logger.info(f"  Exact matches: {exact_hits:,}/{vocab_size:,} ({exact_hits/vocab_size*100:.2f}%)")
        logger.info(f"  Compound words: {compound_hits:,}/{vocab_size:,} ({compound_hits/vocab_size*100:.2f}%)")
        logger.info(f"  Total coverage: {total_coverage:,}/{vocab_size:,} ({total_coverage/vocab_size*100:.2f}%)")
        logger.info(f"  OOV (zeros): {oov_count:,}/{vocab_size:,} ({oov_count/vocab_size*100:.2f}%)")
    
    def _get_compound_embedding(self, token, vocab):
        """Generate embedding for compound words by averaging components."""
        embeddings = []
        
        # Try splitting on common separators
        for separator in ['-', '_', '/']:
            if separator in token:
                parts = token.split(separator)
                for part in parts:
                    if part.lower() in vocab and len(part) > 1:
                        embeddings.append(self.word2vec_model.wv[part.lower()])
        
        # Try prefix/suffix matching for long words
        if len(token) > 6:
            for length in [3, 4, 5]:
                prefix = token[:length].lower()
                suffix = token[-length:].lower()
                
                if prefix in vocab:
                    embeddings.append(self.word2vec_model.wv[prefix])
                if suffix in vocab and suffix != prefix:
                    embeddings.append(self.word2vec_model.wv[suffix])
        
        # Return average if found components
        if embeddings:
            return np.mean(embeddings, axis=0).astype(np.float32)
        else:
            return None
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get Word2Vec embeddings for input token IDs."""
        return self.embedding(input_ids)
    
    def get_coverage_stats(self):
        """Get statistics about vocabulary coverage."""
        vocab = self.tokenizer.get_vocab()
        word2vec_vocab = set(self.word2vec_model.wv.key_to_index.keys())
        
        stats = {
            'biobert_vocab_size': len(vocab),
            'word2vec_vocab_size': len(word2vec_vocab),
            'embedding_dim': self.embedding_dim
        }
        
        return stats

# Simplified interface
class Word2VecEmbedder(FineTunedWord2VecEmbedder):
    """
    Simplified interface that always uses fine-tuning.
    This is the main embedder class to use.
    """
    
    def __init__(self, word2vec_model, tokenizer, training_sentences=None, **kwargs):
        """
        Initialize Word2Vec embedder with automatic fine-tuning.
        
        Args:
            word2vec_model: Path to Word2Vec model or loaded model (for compatibility)
            tokenizer: BioBERT tokenizer
            training_sentences: Training sentences for fine-tuning
            **kwargs: Additional arguments
        """
        if isinstance(word2vec_model, str):
            # Path provided
            pretrained_path = word2vec_model
        else:
            # For backward compatibility, assume it's a path
            raise ValueError("word2vec_model should be a path to the model file")
        
        if training_sentences is None:
            raise ValueError("training_sentences is required for fine-tuning")
        
        super().__init__(
            pretrained_word2vec_path=pretrained_path,
            training_sentences=training_sentences,
            tokenizer=tokenizer,
            **kwargs
        )

# For backward compatibility
ImprovedWord2VecEmbedder = Word2VecEmbedder
