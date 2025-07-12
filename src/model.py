#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model definitions for BioBERT NER with Adaptive Token-Sequence Loss
Fixed Word2Vec integration
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from TorchCRF import CRF

from loss import AdaptiveWeighter, CombinedLoss
from word2vec_embedder import Word2VecEmbedder

class CRFModel(nn.Module):
    def __init__(self, config, dropout_rate=0.3, learning_rate=2e-5, weight_decay=1e-4, 
                 alpha_init=0.5, alpha_lr=0.01, classifier_lr=1e-4):
        super().__init__()
        self.num_labels = config["num_labels"]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha_lr = alpha_lr
        self.classifier_lr = classifier_lr
        self.id_to_label = config["id_to_label"]
        
        # Load BioBERT with memory optimization
        self.bert = AutoModel.from_pretrained(
            config["biobert_model"], 
            output_attentions=False, 
            output_hidden_states=False
        )
        
        # CRF layer - correct initialization for TorchCRF
        self.crf = CRF(self.num_labels)
        
        # Dropout layer
        self.dropout = nn.Dropout(config["dropout_rate"] if "dropout_rate" in config else dropout_rate)
        
        # **SIMPLIFIED**: Always fine-tune Word2Vec on training data, use zeros for remaining OOV
        self.word2vec_embedder = Word2VecEmbedder(
            word2vec_model=config["word2vec_path"],  # Path to pre-trained model
            tokenizer=config["tokenizer"],
            training_sentences=config["training_sentences"],
            finetune_epochs=config.get("word2vec_finetune_epochs", 5),
            min_count=config.get("word2vec_min_count", 1),
            max_vocab_size=config.get("max_vocab_size", 50000),
            cache_dir=config.get("word2vec_cache_dir", "./word2vec_cache")
        )
        self.embedding_dim = self.word2vec_embedder.embedding_dim
        
        # Freeze BERT layers if specified
        freeze_layers = config.get("freeze_bert_layers", 0)
        if freeze_layers > 0:
            # Freeze embeddings layer
            modules_to_freeze = [self.bert.embeddings]
            
            # Freeze specified number of encoder layers
            if hasattr(self.bert, 'encoder') and hasattr(self.bert.encoder, 'layer'):
                num_layers = min(freeze_layers, len(self.bert.encoder.layer))
                modules_to_freeze.extend(self.bert.encoder.layer[:num_layers])
            
            # Apply freezing
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
        
        # Classifier layer - concatenates BERT and Word2Vec embeddings
        self.classifier = nn.Linear(self.bert.config.hidden_size + self.embedding_dim, self.num_labels)
        
        # Combined loss function with adaptive weighting
        self.combined_loss = CombinedLoss(self.crf, alpha_init=alpha_init)

    def forward(self, input_ids, attention_mask=None, labels=None): 
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # **FIXED**: Get proper Word2Vec embeddings with correct vocabulary mapping
        word2vec_embeddings = self.word2vec_embedder(input_ids)  # (batch_size, seq_len, word2vec_dim)
        
        # Concatenate BERT embeddings and Word2Vec embeddings
        combined_embeddings = torch.cat([sequence_output, word2vec_embeddings], dim=-1)  
        combined_embeddings = self.dropout(combined_embeddings)
        logits = self.classifier(combined_embeddings)

        if labels is not None:
            # Return individual components for monitoring
            loss, ce_loss, crf_loss, alpha = self.combined_loss(logits, labels, attention_mask)
            return loss, ce_loss, crf_loss, alpha, logits
        else:
            # For prediction, use CRF decoder
            predictions = self.crf.viterbi_decode(logits, mask=attention_mask.byte())
            return predictions, logits

    def get_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        # Group parameters for different learning rates
        optimizer_grouped_parameters = [
            # BERT parameters with weight decay
            {
                'params': [p for n, p in param_optimizer if 'bert' in n and not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay,
                'lr': self.learning_rate
            },
            # BERT parameters without weight decay
            {
                'params': [p for n, p in param_optimizer if 'bert' in n and any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
                'lr': self.learning_rate
            },
            # Word2Vec embeddings - higher learning rate for fine-tuning
            {
                'params': [p for n, p in param_optimizer if 'word2vec_embedder' in n],
                'weight_decay': self.weight_decay,
                'lr': self.learning_rate * 2
            },
            # Classifier parameters
            {
                'params': [p for n, p in param_optimizer if 'classifier' in n],
                'weight_decay': self.weight_decay,
                'lr': self.classifier_lr
            },
            # CRF parameters
            {
                'params': [p for n, p in param_optimizer if 'crf' in n],
                'weight_decay': self.weight_decay,
                'lr': self.classifier_lr
            },
            # Alpha parameter - with appropriate learning rate for adaptation
            {
                'params': self.combined_loss.adaptive_weighter.parameters(),
                'weight_decay': 0.0,
                'lr': self.alpha_lr
            }
        ]
        
        return torch.optim.AdamW(optimizer_grouped_parameters)
