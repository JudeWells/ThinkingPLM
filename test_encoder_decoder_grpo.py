#!/usr/bin/env python3
"""
Quick test: verify that encoder-decoder GRPO keeps prompt KV cache frozen
while updating decoder likelihoods.

1. Load model, init encoder-decoder GRPO
2. Compute prompt KV cache with frozen encoder — snapshot it
3. Compute decoder log-probs for some completions — snapshot them
4. Run a few GRPO training steps with synthetic rewards
5. Re-compute prompt KV cache — verify unchanged
6. Re-compute decoder log-probs — verify changed
"""

import torch
import copy
from pathlib import Path

from src.models.llama import LlamaLitModule


def main():
    ckpt_dir = Path(".profam_repo/model_checkpoints/profam-1")
    ckpt_path = ckpt_dir / "checkpoints" / "last.ckpt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Loading model (device={device})...")
    ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_obj = ckpt_blob["hyper_parameters"]["config"]
    setattr(cfg_obj, "attn_implementation", "sdpa")
    setattr(cfg_obj, "_attn_implementation", "sdpa")

    model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(
        ckpt_path, config=cfg_obj, strict=False, weights_only=False,
    )
    model.eval()
    model.to(device, dtype=dtype)
    print("Model loaded.\n")

    # ── Set up encoder-decoder GRPO ─────────────────────────────────────
    model.grpo_enabled = True
    model.grpo_normalize_rewards = True
    model.grpo_reward_baseline = "mean"
    model.grpo_clip_ratio = 0.2
    model.grpo_beta = 0.0
    model.grpo_use_reference_model = False
    model.grpo_max_tokens = 8000
    model.init_encoder_decoder_grpo()
    print("Encoder-decoder GRPO initialized.")

    # ── Build a synthetic prompt + completions ──────────────────────────
    tok = model.tokenizer
    sep_id = tok.sep_token_id
    # Use a simple protein sequence as prompt
    prompt_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKA"
    # Tokenize character by character (ProFam uses single-AA tokens)
    prompt_tokens = [tok.convert_tokens_to_ids(aa) for aa in prompt_seq]
    input_ids = torch.tensor([prompt_tokens], device=device)
    print(f"Prompt: {len(prompt_seq)} residues, {input_ids.shape[1]} tokens")

    # Generate a few completions (fake — just random protein-like tokens)
    n_completions = 6
    comp_len = 40
    rng = torch.Generator(device=device).manual_seed(42)
    # Use token IDs in a reasonable range (skip special tokens)
    completion_tokens = torch.randint(
        5, 25, (n_completions, comp_len), device=device, generator=rng,
    )
    # Prepend separator
    sep_col = torch.full((n_completions, 1), sep_id, device=device)
    completion_ids = torch.cat([sep_col, completion_tokens], dim=1)
    completion_ids = completion_ids.unsqueeze(0)  # (1, N, L)
    print(f"Completions: {n_completions} x {comp_len} tokens\n")

    # ── Step 1: Snapshot encoder KV cache ───────────────────────────────
    print("=" * 60)
    print("STEP 1: Snapshot encoder KV cache (frozen)")
    print("=" * 60)
    with torch.no_grad():
        enc_out = model._encoder_model(input_ids=input_ids, use_cache=True)
    kv_cache_before = []
    for layer_kv in enc_out.past_key_values:
        kv_cache_before.append((layer_kv[0].clone(), layer_kv[1].clone()))
    print(f"  KV cache: {len(kv_cache_before)} layers, "
          f"key shape: {kv_cache_before[0][0].shape}")

    # ── Step 2: Snapshot decoder log-probs ──────────────────────────────
    print("\nSTEP 2: Snapshot decoder log-probs")
    model.eval()
    with torch.no_grad():
        lps_before, mask_before = model._compute_per_token_log_probs_for_grpo(
            input_ids=input_ids,
            completion_ids=completion_ids,
        )
    lps_before = lps_before.clone()
    print(f"  Log-probs shape: {lps_before.shape}")
    print(f"  Mean log-prob: {(lps_before * mask_before).sum() / mask_before.sum():.6f}")

    # ── Step 3: Run GRPO training steps ─────────────────────────────────
    print("\nSTEP 3: Run 5 GRPO training steps with synthetic rewards")
    print("=" * 60)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,  # High LR to make changes visible
    )

    # Synthetic rewards: first completion is "best"
    rewards = torch.tensor([1.0, 0.5, 0.2, -0.1, -0.3, -0.5], device=device)

    # We need old log-probs for the importance ratio
    with torch.no_grad():
        old_lps, old_mask = model._compute_per_token_log_probs_for_grpo(
            input_ids=input_ids,
            completion_ids=completion_ids,
        )

    model.train()
    for step in range(5):
        optimizer.zero_grad()
        loss, metrics = model.grpo_step_from_rewards(
            input_ids=input_ids,
            generated_tokens=completion_tokens,
            old_per_token_lps=old_lps,
            old_per_token_mask=old_mask,
            rewards=rewards,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        print(f"  Step {step}: loss={metrics['grpo_loss']:.6f}, "
              f"grad_norm={grad_norm:.4f}, "
              f"clip_frac={metrics['clip_fraction']:.3f}")

    # ── Step 4: Re-check encoder KV cache ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Re-compute encoder KV cache — should be UNCHANGED")
    print("=" * 60)
    with torch.no_grad():
        enc_out_after = model._encoder_model(input_ids=input_ids, use_cache=True)

    all_match = True
    max_diff = 0.0
    for i, (layer_kv_before, layer_kv_after) in enumerate(
        zip(kv_cache_before, enc_out_after.past_key_values)
    ):
        k_diff = (layer_kv_before[0] - layer_kv_after[0]).abs().max().item()
        v_diff = (layer_kv_before[1] - layer_kv_after[1]).abs().max().item()
        layer_max = max(k_diff, v_diff)
        max_diff = max(max_diff, layer_max)
        if layer_max > 1e-6:
            print(f"  Layer {i}: CHANGED! max diff = {layer_max:.2e}")
            all_match = False

    if all_match:
        print(f"  PASS: All layers unchanged (max diff = {max_diff:.2e})")
    else:
        print(f"  FAIL: KV cache changed (max diff = {max_diff:.2e})")

    # ── Step 5: Re-check decoder log-probs ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Re-compute decoder log-probs — should be CHANGED")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        lps_after, mask_after = model._compute_per_token_log_probs_for_grpo(
            input_ids=input_ids,
            completion_ids=completion_ids,
        )

    lp_diff = (lps_before - lps_after).abs()
    valid_diff = lp_diff[mask_before]
    mean_diff = valid_diff.mean().item()
    max_lp_diff = valid_diff.max().item()
    mean_before = (lps_before * mask_before).sum().item() / mask_before.sum().item()
    mean_after = (lps_after * mask_after).sum().item() / mask_after.sum().item()

    if max_lp_diff > 1e-6:
        print(f"  PASS: Decoder log-probs changed")
    else:
        print(f"  FAIL: Decoder log-probs unchanged")
    print(f"  Mean log-prob before: {mean_before:.6f}")
    print(f"  Mean log-prob after:  {mean_after:.6f}")
    print(f"  Mean |diff|: {mean_diff:.6f}")
    print(f"  Max  |diff|: {max_lp_diff:.6f}")

    # ── Also verify decoder model weights changed ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Verify decoder vs encoder weight divergence")
    print("=" * 60)
    n_changed = 0
    n_total = 0
    max_w_diff = 0.0
    for (name, p_dec), p_enc in zip(
        model.model.named_parameters(),
        model._encoder_model.parameters(),
    ):
        n_total += 1
        diff = (p_dec.data - p_enc.data).abs().max().item()
        max_w_diff = max(max_w_diff, diff)
        if diff > 1e-8:
            n_changed += 1

    if n_changed > 0:
        print(f"  PASS: {n_changed}/{n_total} parameter tensors differ "
              f"(max diff = {max_w_diff:.6e})")
    else:
        print(f"  FAIL: No parameter tensors changed")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    kv_ok = all_match
    lp_ok = max_lp_diff > 1e-6
    w_ok = n_changed > 0
    if kv_ok and lp_ok and w_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        if not kv_ok:
            print("  - Encoder KV cache changed (should be frozen)")
        if not lp_ok:
            print("  - Decoder log-probs unchanged (should have updated)")
        if not w_ok:
            print("  - Decoder weights unchanged (should have updated)")
    print("=" * 60)


if __name__ == "__main__":
    main()
