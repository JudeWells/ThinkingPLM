# Thompson Sampling for Conditioning Sequence Selection — Design Decisions

This document records the reasoning behind the design choices made when adding Thompson sampling to the ProFam + BAGEL pipeline.

---

## 1. Why Thompson sampling?

### Problem with the greedy approach

The existing pipeline uses a **greedy** strategy: score all generated sequences by energy, convert to probabilities via softmax, sample an injection set, and optionally apply elitism and conditional swap. This implicitly assumes that the best-scoring sequence is the best one to *condition on* for the next generation cycle.

This assumption can fail in two ways:

1. **Good scores, poor generation**: A sequence may have excellent ipSAE (strong predicted binding) but consistently produce mediocre progeny when used as a ProFam conditioning sequence. This can happen if the sequence sits in a region of sequence space that ProFam cannot easily extend or diversify from.

2. **Mediocre scores, exceptional generation**: A sequence may have middling ipSAE but occasionally produce exceptional progeny — it sits near a productive region of sequence space that ProFam explores well. The greedy approach underweights these sequences.

### Why bandits?

This is a classic **exploration vs exploitation** tradeoff. We want to:
- **Exploit** sequences we know produce good progeny.
- **Explore** sequences we're uncertain about — they might be even better generators.

Multi-armed bandits, specifically Thompson sampling with Beta posteriors, are a natural fit because:
- The reward (binding quality of progeny) is bounded in [0, 1].
- Beta distributions are the conjugate prior for Bernoulli/bounded observations, making updates trivial.
- Thompson sampling automatically balances exploration and exploitation without requiring an explicit exploration parameter (unlike ε-greedy or UCB).

---

## 2. Reward definition

**Decision**: `reward = clamp(-ipSAE, 0, 1)`

### Rationale

- ipSAE values are negative (more negative = better binding). Typical values range from ~0 (no binding) to ~-0.7 (strong binding).
- Negating maps this to [0, 0.7] for typical binders, with 0 meaning no binding and higher meaning better.
- Clamping to [0, 1] ensures the reward is valid for Beta distribution updates (α += reward, β += 1 - reward).
- We use the **raw (unweighted)** ipSAE value, not the weighted energy, because:
  - The reward should reflect the actual binding quality signal, not an arbitrary weight.
  - Other energy terms (size penalty, designability) are useful for steering the greedy approach but would muddy the bandit's learning signal.

### Alternative considered: binary reward

We considered a binary reward (1 if ipSAE below threshold, 0 otherwise), which would make the Beta posterior a true Bernoulli conjugate. We rejected this because:
- It discards magnitude information (ipSAE of -0.7 is much better than -0.3, but both would score 1).
- The continuous reward preserves this gradation while still being valid for Beta updates.

---

## 3. Bootstrap prior: Beta(1 + r, 2 - r)

**Decision**: When a new arm is created, initialize its Beta posterior using its own ipSAE as the first observation.

### Rationale

- A completely uninformative prior (Beta(1, 1) = uniform) would mean every new sequence starts with maximum uncertainty, leading to excessive exploration of poor sequences.
- Using the sequence's own ipSAE as a bootstrap observation gives the posterior a reasonable starting point: sequences with strong ipSAE start with higher expected reward.
- The prior is still weak (effective sample size of 2), so it gets quickly overwhelmed by actual progeny data after 2–3 selections.
- Beta(1 + r, 2 - r) ensures α + β = 3 always, giving every new arm the same prior strength regardless of its reward — only the mean shifts.

### Alternative considered: informative prior from population statistics

We considered setting the prior based on the mean and variance of all observed ipSAE values. We rejected this because:
- It adds complexity for marginal benefit.
- The bootstrap from the arm's own ipSAE is more intuitive and sufficient.

---

## 4. One arm per sequence (not per conditioning event)

**Decision**: Each unique sequence gets one arm. When a progeny is generated, it becomes a new arm (it is a new sequence). The parent arm's posterior is updated based on the best progeny's reward.

### Rationale

- Treating each sequence as a permanent arm with a growing posterior lets us accumulate evidence about its quality as a generator.
- New progeny become new arms because they are genuinely new sequences that may themselves be good generators.
- This creates a growing tree of arms, where lineage (parent_arm_id) is tracked for analysis.

### Why update with only the best progeny?

When multiple progeny are generated per cycle (`profam_num_samples > 1`), we update the parent arm with only the **best** (most negative ipSAE) progeny's reward. This is because:
- We care about the best outcome a conditioning sequence can produce, not the average.
- If a sequence occasionally produces an exceptional binder among mostly mediocre progeny, that's still valuable — the best-progeny update captures this.

---

## 5. Max-seeking variant (thompson_m_samples)

**Decision**: When `thompson_m_samples > 1`, sample m times from each arm's Beta posterior and take the maximum.

### Rationale

- Standard Thompson sampling (m=1) can be slow to explore when there are many arms with similar posteriors.
- The max-seeking variant biases selection toward arms with high variance (under-explored), because high-variance Beta distributions are more likely to produce an extreme sample among m draws.
- This is a known technique from the Thompson sampling literature for accelerating exploration without sacrificing the Bayesian foundation.
- m=1 is the default (standard Thompson), m=3–5 gives moderate exploration bias, m>10 becomes strongly exploratory.

---

## 6. Architecture: single class, minimal coupling

**Decision**: `ThompsonArm` (dataclass) and `ThompsonSampler` (class) are placed in the same file as the pipeline, with a clean `if/else` branch in the main loop.

### Rationale

- The existing codebase is a single-file pipeline (~2400 lines). Adding a separate module would break this convention and add import complexity for minimal benefit.
- The Thompson path is a complete alternative to the greedy selection block — it doesn't share any selection logic with the greedy path. A clean `if/else` split keeps both paths readable and independently modifiable.
- The greedy path is **completely untouched** — no risk of regression for existing campaigns.

### Why not refactor the greedy path into a strategy pattern?

- The greedy path has accumulated significant complexity (elitism, conditional swap, annealing, memory pooling) that is specific to its approach. Abstracting it into a shared interface would require significant refactoring for no immediate benefit.
- The Thompson path has different inputs and outputs (single arm selection vs subset sampling). Forcing both into a common interface would require awkward adapters.
- YAGNI — if a third strategy is added in the future, a strategy pattern refactor would make sense at that point.

---

## 7. Seed evaluation for Thompson

**Decision**: When `selection_strategy == "thompson"`, seed sequences are always evaluated (same as when `elitism` or `accept_only_improvement` is enabled).

### Rationale

- Thompson sampling needs initial arms to start. Without evaluating seeds, there would be no arms to select from in cycle 1.
- The seed evaluation block already existed for elitism/conditional-swap. We simply added `selection_strategy == "thompson"` to the condition, reusing the same evaluation infrastructure.
- Seed sequences are registered as arms with `parent_arm_id=None` and `created_at_cycle=0`.

---

## 8. Handling folding failures

**Decision**: Sequences with infinite (non-finite) ipSAE are **not** registered as arms and do not update any arm's posterior.

### Rationale

- A folding failure (producing inf energy) is not informative about the parent arm's quality as a generator — it's a failure of the folding oracle, not of the sequence.
- Registering failed sequences as arms with zero reward would unfairly penalize the parent and pollute the arm pool with unscoreable sequences.
- Not updating the parent on failure is conservative: we simply don't learn anything from that cycle about the parent's quality. The parent's posterior remains unchanged, and it may be selected again.

---

## 9. Injection set in Thompson mode

**Decision**: In Thompson mode, the injection set is always a single sequence (the selected arm).

### Rationale

- Thompson sampling selects one arm per cycle — that's the fundamental unit of the bandit algorithm.
- Setting `injected_seqs = [next_arm.sequence]` means ProFam is conditioned on exactly one sequence (plus initial sequences if `reinject_initial` is true).
- This is different from the greedy path, which injects `floor(f_inject * N)` sequences. The Thompson approach is more focused: one conditioning sequence, potentially multiple progeny.
- The `profam_num_samples` parameter still controls how many progeny are generated per cycle. Setting it to 1 gives the purest bandit behavior (one arm → one progeny → one observation). Setting it higher generates more progeny per arm selection, with the best one used for the posterior update.

---

## 10. Logging and observability

**Decision**: Thompson state is logged in three places:
1. `thompson_arms.json` — full arm state, overwritten each cycle.
2. `cycle_stats.json` — per-cycle Thompson-specific fields (selected arm ID, progeny reward, number of arms).
3. Console output — arm selection and posterior update details.

### Rationale

- `thompson_arms.json` enables post-hoc analysis of arm posteriors, lineage trees, and selection patterns.
- `cycle_stats.json` integration means existing plotting tools can read Thompson runs without modification.
- The full arm state (α, β, sequence, parent, creation cycle, selection count) is everything needed to reconstruct the algorithm's decision-making process.

---

## 11. Backward compatibility

**Decision**: All new config fields have backward-compatible defaults (`selection_strategy: "greedy"`, `thompson_m_samples: 1`, `thompson_reward_term: "ipSAE"`).

### Rationale

- Existing YAML configs that don't mention these fields continue to work identically.
- The greedy code path is completely unchanged — no risk of regression.
- The `selection_strategy` field is validated at config load time, so typos are caught early.
