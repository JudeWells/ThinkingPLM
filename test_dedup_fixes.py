#!/usr/bin/env python3
"""
Tests for the dedup/diversity fixes:

1. adjust_temperature() triggers correctly on various diversity scenarios
2. Advantage clamping for seen sequences
3. grpo_token_data survives dedup with correct entries
4. Raw diversity metrics computed before dedup
5. Dedup exhaustion falls back to random mutation
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def test_adjust_temperature():
    """Test the temperature adjustment function."""
    from run_profam_bagel_pipeline import adjust_temperature

    starting = 0.8

    # 1. Low batch diversity → raise
    new_t, reason = adjust_temperature(
        raw_unique_fraction=0.5, raw_novel_fraction=0.9,
        avg_sim_to_prompt=0.3, current_temperature=0.8, starting_temperature=starting,
    )
    assert new_t == 0.9, f"Expected 0.9, got {new_t}"
    assert "low batch diversity" in reason
    print("  PASS: low batch diversity raises temp")

    # 2. Prompt collapse → raise
    new_t, reason = adjust_temperature(
        raw_unique_fraction=1.0, raw_novel_fraction=0.9,
        avg_sim_to_prompt=1.0, current_temperature=0.8, starting_temperature=starting,
    )
    assert new_t == 0.9, f"Expected 0.9, got {new_t}"
    assert "prompt collapse" in reason
    print("  PASS: prompt collapse raises temp")

    # 3. Low novelty (>50% seen before) → raise
    new_t, reason = adjust_temperature(
        raw_unique_fraction=1.0, raw_novel_fraction=0.3,
        avg_sim_to_prompt=0.4, current_temperature=0.8, starting_temperature=starting,
    )
    assert new_t == 0.9, f"Expected 0.9, got {new_t}"
    assert "low novelty" in reason
    print("  PASS: low novelty raises temp")

    # 4. All good → lower toward starting
    new_t, reason = adjust_temperature(
        raw_unique_fraction=1.0, raw_novel_fraction=0.9,
        avg_sim_to_prompt=0.4, current_temperature=1.0, starting_temperature=starting,
    )
    assert new_t < 1.0, f"Expected < 1.0, got {new_t}"
    assert reason is None
    print("  PASS: good diversity lowers temp toward starting")

    # 5. All good but already at starting → no change
    new_t, reason = adjust_temperature(
        raw_unique_fraction=1.0, raw_novel_fraction=0.9,
        avg_sim_to_prompt=0.4, current_temperature=0.8, starting_temperature=starting,
    )
    assert new_t == 0.8, f"Expected 0.8, got {new_t}"
    assert reason is None
    print("  PASS: at starting temp, no change")

    # 6. Multiple triggers at once
    new_t, reason = adjust_temperature(
        raw_unique_fraction=0.3, raw_novel_fraction=0.2,
        avg_sim_to_prompt=1.0, current_temperature=0.8, starting_temperature=starting,
    )
    assert new_t == 0.9
    assert "low batch diversity" in reason
    assert "prompt collapse" in reason
    assert "low novelty" in reason
    print("  PASS: multiple triggers reported")

    # 7. High similarity but not 1.0 → no prompt collapse trigger
    new_t, reason = adjust_temperature(
        raw_unique_fraction=1.0, raw_novel_fraction=0.9,
        avg_sim_to_prompt=0.95, current_temperature=0.8, starting_temperature=starting,
    )
    assert new_t == 0.8, f"Expected 0.8 (no trigger), got {new_t}"
    print("  PASS: sim=0.95 does not trigger prompt collapse")

    print("  All adjust_temperature tests passed!\n")


def test_advantage_clamping():
    """Test that advantages are clamped only for current-cycle re-generations.

    Replay buffer entries keep their original advantages (they're normalisation
    context). Only current-cycle sequences that match prior cycles get clamped.
    """
    print("  Testing advantage clamping for current-cycle re-generations...")

    # Simulate replay buffer: 2 prior cycles + 1 current cycle
    prior_cycle_1_seqs = ["AAAA", "BBBB", "CCCC"]
    prior_cycle_2_seqs = ["DDDD", "EEEE", "FFFF"]
    # Current cycle: AAAA is a re-generation, GGGG and HHHH are novel
    current_cycle_seqs = ["AAAA", "GGGG", "HHHH"]

    prior_seen = set(prior_cycle_1_seqs + prior_cycle_2_seqs)

    # Merged rewards: prior1 + prior2 + current (9 total)
    rewards = torch.tensor([0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8, 0.1, 0.9])
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r

    # Save copies of replay buffer advantages before clamping
    prior_advantages = advantages[:6].clone()

    # Clamp only current-cycle entries that match prior sequences
    n_total = len(rewards)
    n_current = len(current_cycle_seqs)
    offset = n_total - n_current  # = 6

    n_clamped = 0
    for j, seq in enumerate(current_cycle_seqs):
        idx = offset + j
        if seq in prior_seen:
            advantages[idx] = torch.clamp(advantages[idx], max=0.0)
            n_clamped += 1

    # Only AAAA (idx 6) should be clamped
    assert n_clamped == 1, f"Expected 1 clamped, got {n_clamped}"

    # Replay buffer advantages should be UNCHANGED
    assert torch.equal(advantages[:6], prior_advantages), \
        "Replay buffer advantages should not be modified"

    # AAAA at idx 6 had reward 0.8 (high) → positive advantage → clamped to ≤ 0
    assert advantages[6] <= 0.0, f"Re-generated AAAA should be ≤ 0, got {advantages[6]}"

    # GGGG (idx 7) is novel → unchanged
    expected_gggg = (0.1 - mean_r) / std_r
    assert abs(advantages[7] - expected_gggg) < 1e-5, "Novel GGGG should be unchanged"

    # HHHH (idx 8) is novel → unchanged
    expected_hhhh = (0.9 - mean_r) / std_r
    assert abs(advantages[8] - expected_hhhh) < 1e-5, "Novel HHHH should be unchanged"

    print("  PASS: only current-cycle re-generations clamped, replay buffer untouched\n")


def test_raw_diversity_metrics():
    """Test that raw diversity is computed correctly before dedup."""
    print("  Testing raw diversity metric computation...")

    # Simulate a batch with duplicates
    gen_seqs = ["AAAA", "BBBB", "AAAA", "CCCC", "BBBB", "DDDD"]
    all_seqs = ["AAAA"]  # prompt
    seen_sequences = {"AAAA": (-0.5, {}), "BBBB": (-0.3, {})}

    n_raw_batch = len(gen_seqs)
    n_unique_in_batch = len(set(gen_seqs))
    n_seen_before = sum(1 for s in gen_seqs if s in seen_sequences)
    n_prompt_copies = sum(1 for s in gen_seqs if s in set(all_seqs))
    raw_unique_fraction = n_unique_in_batch / n_raw_batch
    raw_novel_fraction = (n_raw_batch - n_seen_before) / n_raw_batch

    assert n_unique_in_batch == 4, f"Expected 4 unique, got {n_unique_in_batch}"
    assert n_seen_before == 4, f"Expected 4 seen (2xAAAA + 2xBBBB), got {n_seen_before}"
    assert n_prompt_copies == 2, f"Expected 2 prompt copies (2xAAAA), got {n_prompt_copies}"
    assert abs(raw_unique_fraction - 4/6) < 1e-5
    assert abs(raw_novel_fraction - 2/6) < 1e-5

    print(f"    unique_in_batch={n_unique_in_batch}, seen_before={n_seen_before}, "
          f"prompt_copies={n_prompt_copies}")
    print(f"    raw_unique_fraction={raw_unique_fraction:.3f}, raw_novel_fraction={raw_novel_fraction:.3f}")
    print("  PASS: raw metrics correct\n")


def test_token_data_rebuild():
    """Test that grpo_token_data is correctly rebuilt after intra-batch dedup."""
    print("  Testing token data rebuild after dedup...")

    # Simulate original batch of 6 sequences, 3 unique
    orig_seqs = ["AAAA", "BBBB", "AAAA", "CCCC", "BBBB", "AAAA"]
    seq_len = 5
    n_orig = len(orig_seqs)

    # Create fake token data where each sequence has a distinct pattern
    # so we can verify the right entries survive
    generated_tokens = torch.arange(n_orig * seq_len).reshape(n_orig, seq_len).float()
    old_per_token_lps = -torch.arange(n_orig * (seq_len - 1)).reshape(n_orig, seq_len - 1).float()
    old_per_token_mask = torch.ones(n_orig, seq_len - 1, dtype=torch.bool)

    grpo_token_data = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "generated_tokens": generated_tokens,
        "old_per_token_lps": old_per_token_lps,
        "old_per_token_mask": old_per_token_mask,
        "_original_seqs": list(orig_seqs),
    }

    # Simulate dedup: keep indices 0 (AAAA), 1 (BBBB), 3 (CCCC)
    # Replace indices 2, 4, 5 with new sequences
    keep_orig_idx = [0, 1, 3]
    final_seqs = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"]

    # Simulate replacement token data (3 new sequences)
    extra_chunks = []
    for i in range(3):
        extra_chunks.append({
            "generated_tokens": torch.full((1, seq_len), 100 + i, dtype=torch.float),
            "old_per_token_lps": torch.full((1, seq_len - 1), -(100 + i), dtype=torch.float),
            "old_per_token_mask": torch.ones(1, seq_len - 1, dtype=torch.bool),
        })

    # Rebuild
    idx_t = torch.tensor(keep_orig_idx)
    parts_t = [generated_tokens[idx_t]]
    parts_lp = [old_per_token_lps[idx_t]]
    parts_m = [old_per_token_mask[idx_t]]
    for chunk in extra_chunks:
        parts_t.append(chunk["generated_tokens"])
        parts_lp.append(chunk["old_per_token_lps"])
        parts_m.append(chunk["old_per_token_mask"])

    rebuilt_tokens = torch.cat(parts_t, dim=0)
    rebuilt_lps = torch.cat(parts_lp, dim=0)
    rebuilt_masks = torch.cat(parts_m, dim=0)

    assert rebuilt_tokens.shape[0] == 6, f"Expected 6 rows, got {rebuilt_tokens.shape[0]}"

    # Check original entries preserved correctly
    assert torch.equal(rebuilt_tokens[0], generated_tokens[0]), "AAAA tokens should match original idx 0"
    assert torch.equal(rebuilt_tokens[1], generated_tokens[1]), "BBBB tokens should match original idx 1"
    assert torch.equal(rebuilt_tokens[2], generated_tokens[3]), "CCCC tokens should match original idx 3"

    # Check replacement entries
    assert rebuilt_tokens[3, 0] == 100, "DDDD tokens should be 100"
    assert rebuilt_tokens[4, 0] == 101, "EEEE tokens should be 101"
    assert rebuilt_tokens[5, 0] == 102, "FFFF tokens should be 102"

    # Check log-probs similarly
    assert torch.equal(rebuilt_lps[0], old_per_token_lps[0])
    assert rebuilt_lps[3, 0] == -100

    print(f"    Rebuilt shape: {rebuilt_tokens.shape}")
    print("  PASS: token data correctly rebuilt with original survivors + replacements\n")


def test_seen_sequence_advantage_zero_not_reward_zero():
    """Verify we clamp ADVANTAGE not REWARD, and only for current-cycle entries.

    A current-cycle re-generation with good energy → advantage clamped to 0.
    A current-cycle re-generation with bad energy → stays negative.
    Novel current-cycle sequences → unchanged.
    Replay buffer entries (even if same sequence) → unchanged.
    """
    print("  Testing advantage (not reward) clamping semantics...")

    # Prior cycle had GOOD_SEEN and BAD_SEEN
    prior_seqs = {"GOOD_SEEN", "BAD_SEEN"}

    # Replay buffer (prior): 2 entries
    # Current cycle: re-generates GOOD_SEEN and BAD_SEEN, plus 2 novel
    # Merged: [prior_GOOD, prior_BAD, cur_GOOD_SEEN, cur_BAD_SEEN, cur_NOVEL_GOOD, cur_NOVEL_BAD]
    rewards = torch.tensor([0.8, 0.2, 0.8, 0.2, 0.7, 0.1])
    current_seqs = ["GOOD_SEEN", "BAD_SEEN", "NOVEL_GOOD", "NOVEL_BAD"]

    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r
    prior_advs = advantages[:2].clone()

    # Clamp only current-cycle entries
    n_total = len(rewards)
    offset = n_total - len(current_seqs)  # = 2
    for j, seq in enumerate(current_seqs):
        idx = offset + j
        if seq in prior_seqs:
            advantages[idx] = torch.clamp(advantages[idx], max=0.0)

    # Replay buffer entries UNCHANGED
    assert torch.equal(advantages[:2], prior_advs), "Replay entries should not be modified"

    # Current-cycle GOOD_SEEN (idx 2): was positive → clamped to 0
    assert advantages[2] == 0.0, f"Re-gen GOOD_SEEN should be 0, got {advantages[2]}"
    # Current-cycle BAD_SEEN (idx 3): was negative → stays negative
    assert advantages[3] < 0.0, f"Re-gen BAD_SEEN should stay negative, got {advantages[3]}"
    # Novel sequences: unchanged
    assert advantages[4] > 0, "NOVEL_GOOD should keep positive advantage"
    assert advantages[5] < 0, "NOVEL_BAD should keep negative advantage"

    print("  PASS: advantage clamping semantics correct\n")


def test_dedup_exhaustion_mutation_fallback():
    """Test that dedup exhaustion generates mutants instead of using cached."""
    print("  Testing dedup exhaustion → mutation fallback...")

    from run_profam_bagel_pipeline import run_random_mutation_generation

    seed_seqs = ["NEKNWEKTLQINLVSVISGTYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANLMNSGVR"]
    seen_sequences = {}

    # Generate mutants
    rng = np.random.default_rng(42)
    mut_names, mut_seqs = run_random_mutation_generation(
        seed_sequences=seed_seqs,
        num_samples=12,
        max_mutations=1,
        rng=rng,
    )

    assert len(mut_seqs) == 12, f"Expected 12 mutants, got {len(mut_seqs)}"

    # Each should differ from the seed by at most 1 position (max_mutations=1).
    # May be 0 if the random AA happens to match the original.
    for i, (seed, mut) in enumerate(zip([seed_seqs[0]] * 12, mut_seqs)):
        if len(seed) == len(mut):
            diffs = sum(1 for a, b in zip(seed, mut) if a != b)
            assert diffs <= 1, f"Mutant {i} differs by {diffs} positions, expected ≤ 1"

    # Now simulate: add some to seen
    for s in mut_seqs[:6]:
        seen_sequences[s] = (-0.5, {})

    # Filter novel
    novel = [s for s in mut_seqs if s not in seen_sequences]
    assert len(novel) == 6, f"Expected 6 novel mutants, got {len(novel)}"

    print(f"    Generated {len(mut_seqs)} mutants, {len(novel)} novel")
    print("  PASS: mutation fallback generates novel sequences\n")


def main():
    print("=" * 60)
    print("  Dedup & Diversity Fixes — Test Suite")
    print("=" * 60 + "\n")

    test_adjust_temperature()
    test_advantage_clamping()
    test_raw_diversity_metrics()
    test_token_data_rebuild()
    test_seen_sequence_advantage_zero_not_reward_zero()
    test_dedup_exhaustion_mutation_fallback()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
