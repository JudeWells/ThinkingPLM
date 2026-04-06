#!/usr/bin/env python3
"""
Synthetic reward tests for GRPO: verify the training pipeline can shift
the model's output distribution toward higher-reward sequences.

Test 1 — Copy reward: reward = sequence identity to the prompt.
  If GRPO works, generated sequences should become more similar to the prompt.

Test 2 — Shift reward: reward each position where the residue is the "next"
  amino acid (A→C, C→D, D→E, ..., Y→A).  If GRPO works, the model should
  learn to systematically shift each residue by one position in the alphabet.

Both tests bypass the pipeline entirely: load model → generate → compute
synthetic reward → call grpo_step_from_rewards → repeat.

Usage:
    python test_grpo_synthetic.py --test copy --steps 100
    python test_grpo_synthetic.py --test shift --steps 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Amino acid alphabet (canonical 20)
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
# Shift map: each AA maps to the next one in the alphabet, Y wraps to A
SHIFT_MAP = {AA[i]: AA[(i + 1) % len(AA)] for i in range(len(AA))}


def _aligned_identity(seq: str, target: str) -> float:
    """Sequence identity via pairwise alignment, normalised by max length."""
    from Bio.Align import PairwiseAligner
    if not seq or not target:
        return 0.0
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    alignment = aligner.align(seq, target)[0]
    matches = sum(
        1 for a, b in zip(alignment.target, alignment.query)
        if a == b and a != "-"
    )
    return matches / max(len(seq), len(target))


def compute_copy_reward(sequences: list[str], prompt_seq: str) -> np.ndarray:
    """Reward = aligned sequence identity to the prompt."""
    return np.array(
        [_aligned_identity(seq, prompt_seq) for seq in sequences],
        dtype=np.float32,
    )


def compute_shift_reward(sequences: list[str], prompt_seq: str) -> np.ndarray:
    """Reward = aligned identity to the shifted prompt (each AA shifted +1)."""
    target = "".join(SHIFT_MAP.get(aa, aa) for aa in prompt_seq)
    return np.array(
        [_aligned_identity(seq, target) for seq in sequences],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description="Synthetic GRPO test")
    parser.add_argument("--test", choices=["copy", "shift"], default="copy")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--prompt", type=str,
                        default="NEKNWEKTLQINLVSVISGTYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANLMNSGVR")
    parser.add_argument("--frozen_encoder", action="store_true",
                        help="Use frozen encoder for prompt KV cache")
    args = parser.parse_args()

    reward_fn = compute_copy_reward if args.test == "copy" else compute_shift_reward
    target_label = "prompt" if args.test == "copy" else "shifted prompt"
    if args.test == "shift":
        target_seq = "".join(SHIFT_MAP.get(aa, aa) for aa in args.prompt)
        print(f"Shift target: {target_seq[:40]}...")
    else:
        target_seq = args.prompt

    # Load model
    from run_profam_bagel_pipeline import load_profam_model, PipelineConfig

    cfg = PipelineConfig(
        initial_fasta=None,
        profam_checkpoint_dir=Path(".profam_repo/model_checkpoints/profam-1"),
        energy_config=Path("configs/energy/energy_lis_2GDZ_local.yaml"),
        random_init=True,
    )
    model, device = load_profam_model(cfg)
    tok = model.tokenizer

    # GRPO config
    model.grpo_enabled = True
    model.grpo_normalize_rewards = True
    model.grpo_reward_baseline = "mean"
    model.grpo_clip_ratio = args.clip_ratio
    model.grpo_beta = 0.0
    model.grpo_use_reference_model = False
    model.grpo_max_tokens = 8000
    if args.frozen_encoder:
        model.init_encoder_decoder_grpo()
        print("Encoder-decoder GRPO initialized (frozen encoder, trainable decoder)")
    else:
        print("Standard GRPO initialized (no frozen encoder)")

    # Build prompt tokens: [BOS] [RAW] sequence [SEP]
    # [SEP] marks end of a sequence — the model then generates the next family member
    prompt_token_ids = [tok.bos_token_id, tok.convert_tokens_to_ids("[RAW]")]
    for aa in args.prompt:
        prompt_token_ids.append(tok.convert_tokens_to_ids(aa))
    prompt_token_ids.append(tok.sep_token_id)
    input_ids = torch.tensor([prompt_token_ids], device=device)

    # Optimizer — only trainable params (excludes frozen encoder)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Baseline: generate before any training
    model.eval()
    with torch.no_grad():
        tokens, scores, _, _ = model._sample_seqs(
            input_ids, num_samples=args.group_size, max_tokens=8000,
            return_per_token_log_probs=True, max_retries=0,
        )
    seqs = model.tokenizer.decode_tokens(tokens.to(device))
    if isinstance(seqs[0], list):
        seqs = [s[0] if s else "" for s in seqs]
    baseline_rewards = reward_fn(seqs, args.prompt)
    print(f"\nBaseline sequences (first 3):")
    for i in range(min(3, len(seqs))):
        print(f"  [{i}] len={len(seqs[i])}: {seqs[i][:80]}...")
        print(f"       reward={baseline_rewards[i]:.4f}")
    print(f"  Prompt:             {args.prompt[:80]}")
    print(f"\n{'=' * 60}")
    print(f"  GRPO Synthetic Test: {args.test}")
    print(f"  Prompt:  {args.prompt[:50]}...")
    print(f"  Target:  {target_seq[:50]}...")
    print(f"  Steps:   {args.steps}, group_size={args.group_size}, lr={args.lr}")
    print(f"{'=' * 60}")
    print(f"\nBaseline (before training):")
    print(f"  Mean reward: {baseline_rewards.mean():.4f}")
    print(f"  Max reward:  {baseline_rewards.max():.4f}")
    print(f"  Example seq: {seqs[0][:50]}...")

    # Training loop
    history = {
        "step": [], "mean_reward": [], "max_reward": [],
        "grpo_loss": [], "clip_fraction": [], "grad_norm": [],
    }

    for step in range(1, args.steps + 1):
        t0 = time.time()

        # Generate
        model.train()
        with torch.no_grad():
            tokens, scores, old_lps, old_mask = model._sample_seqs(
                input_ids, num_samples=args.group_size, max_tokens=8000,
                return_per_token_log_probs=True, max_retries=0,
            )

        # Decode
        seqs = model.tokenizer.decode_tokens(tokens.to(device))
        if isinstance(seqs[0], list):
            seqs = [s[0] if s else "" for s in seqs]

        # Compute synthetic reward
        rewards_np = reward_fn(seqs, args.prompt)
        rewards = torch.tensor(rewards_np, device=device)

        # GRPO step
        total_loss, metrics = model.grpo_step_from_rewards(
            input_ids=input_ids,
            generated_tokens=tokens,
            old_per_token_lps=old_lps,
            old_per_token_mask=old_mask,
            rewards=rewards,
            clip_ratio=args.clip_ratio,
        )

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        elapsed = time.time() - t0

        # Log
        history["step"].append(step)
        history["mean_reward"].append(float(rewards_np.mean()))
        history["max_reward"].append(float(rewards_np.max()))
        history["grpo_loss"].append(metrics["grpo_loss"])
        history["clip_fraction"].append(metrics["clip_fraction"])
        history["grad_norm"].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm))

        if step % 10 == 0 or step == 1:
            best_idx = int(np.argmax(rewards_np))
            print(
                f"  Step {step:>4}: mean_r={rewards_np.mean():.4f}, "
                f"max_r={rewards_np.max():.4f}, "
                f"loss={metrics['grpo_loss']:.4f}, "
                f"clip={metrics['clip_fraction']:.3f}, "
                f"gnorm={history['grad_norm'][-1]:.2f}, "
                f"t={elapsed:.1f}s"
            )
            print(f"           best: {seqs[best_idx][:60]}...")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        tokens, scores, _, _ = model._sample_seqs(
            input_ids, num_samples=args.group_size, max_tokens=8000,
            return_per_token_log_probs=True, max_retries=0,
        )
    seqs = model.tokenizer.decode_tokens(tokens.to(device))
    if isinstance(seqs[0], list):
        seqs = [s[0] if s else "" for s in seqs]
    final_rewards = reward_fn(seqs, args.prompt)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {args.test} test")
    print(f"{'=' * 60}")
    print(f"  Baseline mean reward: {baseline_rewards.mean():.4f}")
    print(f"  Final mean reward:    {final_rewards.mean():.4f}")
    print(f"  Improvement:          {final_rewards.mean() - baseline_rewards.mean():+.4f}")
    print(f"  Baseline max reward:  {baseline_rewards.max():.4f}")
    print(f"  Final max reward:     {final_rewards.max():.4f}")
    best_idx = int(np.argmax(final_rewards))
    print(f"\n  Best final sequence ({args.test} reward={final_rewards[best_idx]:.4f}):")
    print(f"    {seqs[best_idx]}")
    print(f"  Target:")
    print(f"    {target_seq}")

    # Save plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor("black")
        for ax in axes.flat:
            ax.set_facecolor("black")

        axes[0, 0].plot(history["step"], history["mean_reward"], color="#00bfff", label="mean")
        axes[0, 0].plot(history["step"], history["max_reward"], color="#00e676", label="max", alpha=0.7)
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title(f"{args.test.title()} Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.2)

        axes[0, 1].plot(history["step"], history["grpo_loss"], color="#ff6b6b")
        axes[0, 1].set_ylabel("GRPO Loss")
        axes[0, 1].set_title("Loss")
        axes[0, 1].grid(alpha=0.2)

        axes[1, 0].plot(history["step"], history["clip_fraction"], color="#ffab40")
        axes[1, 0].set_ylabel("Clip Fraction")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_title("PPO Clip Fraction")
        axes[1, 0].grid(alpha=0.2)

        axes[1, 1].plot(history["step"], history["grad_norm"], color="#e040fb")
        axes[1, 1].set_ylabel("Grad Norm")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_title("Gradient Norm")
        axes[1, 1].grid(alpha=0.2)

        fig.suptitle(f"GRPO Synthetic Test: {args.test} (lr={args.lr}, group={args.group_size})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path = f"outputs/grpo_synthetic_test_{args.test}.png"
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, facecolor="black", edgecolor="none")
        plt.close(fig)
        print(f"\n  Plot saved to {plot_path}")
    except Exception as e:
        print(f"\n  Plot failed: {e}")


if __name__ == "__main__":
    main()
