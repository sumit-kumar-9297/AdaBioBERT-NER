#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive loss implementation for BioBERT NER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeighter(nn.Module):
    """
    Implements a learnable parameter alpha for adaptive weighting between CE and CRF losses.
    
    The alpha parameter is constrained to [0, 1] range using a sigmoid function.
    Loss is computed as: alpha * ce_loss + (1 - alpha) * crf_loss
    """
    def __init__(self, alpha_init=0.5):
        super().__init__()
        # Initialize alpha_logit to value that maps to alpha_init after sigmoid
        self.alpha_logit = nn.Parameter(torch.tensor(torch.logit(torch.tensor(alpha_init)), dtype=torch.float))
        
    def forward(self, ce_loss, crf_loss):
        # Ensure alpha stays between 0 and 1 using sigmoid
        alpha = torch.sigmoid(self.alpha_logit)
        
        # Compute weighted loss according to the algorithm
        # L_ATS = α · L_CE + (1 - α) · L_CRF
        total_loss = alpha * ce_loss + (1 - alpha) * crf_loss
        
        return total_loss, alpha

class CombinedLoss(nn.Module):
    """
    Combines token-level cross-entropy loss with sequence-level CRF loss using adaptive weighting.
    
    Follows the algorithm:
    1. Compute token-level cross-entropy loss
    2. Compute CRF sequence-level loss
    3. Compute weighted loss as alpha * ce_loss + (1 - alpha) * crf_loss
    4. Update alpha via backpropagation
    """
    def __init__(self, crf, alpha_init=0.5):
        super().__init__()
        self.crf = crf
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.adaptive_weighter = AdaptiveWeighter(alpha_init=alpha_init)

    def forward(self, logits, labels, attention_mask):
        # Calculate token-level cross-entropy loss (L_CE) as in algorithm step 5
        active_tokens = attention_mask.view(-1) == 1
        logits_view = logits.view(-1, logits.size(-1))
        labels_view = labels.view(-1)
        
        # Apply mask for CE loss
        active_mask = (labels_view != -100) & active_tokens
        if active_mask.sum() > 0:  # Check if there are any valid tokens
            masked_logits = logits_view[active_mask]
            masked_labels = labels_view[active_mask]
            ce_loss = self.ce_loss(masked_logits, masked_labels)
        else:
            ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Modify labels for CRF - replace -100 with valid values
        crf_labels = labels.clone()
        invalid_mask = crf_labels == -100
        if invalid_mask.any():
            crf_labels[invalid_mask] = 0
        
        # Calculate CRF sequence-level loss (L_CRF) as in algorithm steps 7-8
        # This computes -log P(Y|X) = -(S(Y,X) - Z(X))
        # Where S(Y,X) is the score for the true sequence and Z(X) is the partition function
        crf_loss = -self.crf(logits, crf_labels, mask=attention_mask.byte())
        
        # Ensure the loss is a scalar
        if crf_loss.dim() > 0:
            crf_loss = crf_loss.mean()
        
        # Apply adaptive weighting according to algorithm step 10
        # L_ATS = α · L_CE + (1 - α) · L_CRF
        total_loss, alpha = self.adaptive_weighter(ce_loss, crf_loss)
            
        return total_loss, ce_loss, crf_loss, alpha