"""Bradley-Terry pairwise ranking loss for online binder design.

Adapted from pref_opt_profam/ranking_ft for use in the pipeline loop.
Uses the frozen encoder KV-cache pattern: prompt context is computed once
with a frozen model copy, then the trainable decoder is updated to rank
sequences by energy (lower energy = higher log-likelihood).
"""

import torch
import torch.nn.functional as F

from src.models.utils import InputAwareDynamicCache, log_likelihood_from_outputs


def bradley_terry_loss(scores, fitness_values):
    """Bradley-Terry pairwise ranking loss.

    Args:
        scores: (B,) differentiable log-likelihood scores from the model
        fitness_values: (B,) fitness values (higher = better, i.e. negated energy)

    Returns:
        Scalar loss tensor with gradients.
    """
    B = scores.shape[0]
    score_diffs = scores.unsqueeze(1) - scores.unsqueeze(0)
    targets = (fitness_values.unsqueeze(1) > fitness_values.unsqueeze(0)).float()
    mask = 1.0 - torch.eye(B, device=scores.device)
    loss = F.binary_cross_entropy_with_logits(score_diffs, targets, reduction="none")
    return 0.5 * (loss * mask).sum() / mask.sum()


def score_variants_differentiable(model, past_key_values, completion_ids, sub_batch_size=4):
    """Score variant sequences with gradient flow through the decoder.

    Args:
        model: LlamaLitModule with .model and .tokenizer.
        past_key_values: Frozen KV cache from the encoder.
        completion_ids: (1, N, L) tensor of tokenized sequences.
        sub_batch_size: Sequences per forward pass.

    Returns:
        scores: (N,) tensor of mean log-likelihoods with gradients.
    """
    pad_token_id = model.tokenizer.pad_token_id
    N = completion_ids.shape[1]
    L = completion_ids.shape[2]
    all_scores = []

    for batch_start in range(0, N, sub_batch_size):
        batch_end = min(batch_start + sub_batch_size, N)
        this_ids = completion_ids[0, batch_start:batch_end]
        actual_bs = this_ids.shape[0]

        # Trim trailing padding
        mask = this_ids != pad_token_id
        indices = torch.arange(L, device=this_ids.device).expand(actual_bs, -1)
        indices = torch.where(mask, indices, torch.zeros_like(indices))
        max_len = indices.max().item() + 1
        this_ids = this_ids[:, :max_len]

        # Expand frozen KV cache
        cache = InputAwareDynamicCache.from_legacy_cache(past_key_values)
        cache.batch_repeat_interleave(actual_bs)

        # Forward pass WITH gradients through decoder
        outputs = model.model(
            input_ids=this_ids,
            past_key_values=cache,
            use_cache=False,
        )

        labels = torch.where(this_ids == pad_token_id, -100, this_ids.clone())
        log_ll = log_likelihood_from_outputs(outputs, labels, start_ix=0)

        shift_labels = labels[..., 1:].to(log_ll.device)
        valid = shift_labels != -100
        denom = valid.sum(dim=-1).clamp(min=1)
        ll_mean = (log_ll * valid).sum(dim=-1) / denom
        all_scores.append(ll_mean)

    return torch.cat(all_scores)
