"""Proposal generation for the ProFam + BAGEL pipeline.

Contains:
- ProposalGenerator ABC
- ProFamProposalGenerator: Uses ProFam language model
- RandomMutationProposalGenerator: Uses random amino acid substitutions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class ProposalGenerator(ABC):
    """Abstract base class for sequence proposal methods."""

    @abstractmethod
    def generate(
        self,
        prompt_seqs: List[str],
        prompt_names: List[str],
        num_samples: int,
        cycle_dir: Path,
    ) -> Tuple[List[str], List[str]]:
        """Generate candidate sequences from prompt sequences.

        Args:
            prompt_seqs: List of prompt sequences to condition on.
            prompt_names: Names corresponding to prompt sequences.
            num_samples: Number of candidate sequences to generate.
            cycle_dir: Directory for this cycle's outputs.

        Returns:
            Tuple of (names, sequences) for generated candidates.
        """
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of this proposal method."""
        pass


class ProFamProposalGenerator(ProposalGenerator):
    """Generates sequences using the ProFam language model."""

    def __init__(
        self,
        model: Any,
        device: str,
        cfg: Any,
    ):
        """Initialize the ProFam generator.

        Args:
            model: The loaded ProFam model.
            device: Device to run the model on.
            cfg: Pipeline configuration object.
        """
        self.model = model
        self.device = device
        self.cfg = cfg

    def generate(
        self,
        prompt_seqs: List[str],
        prompt_names: List[str],
        num_samples: int,
        cycle_dir: Path,
    ) -> Tuple[List[str], List[str]]:
        # Import here to avoid circular imports
        from src.data.objects import ProteinDocument
        from src.data.processors.preprocessing import (
            AlignedProteinPreprocessingConfig,
            ProteinDocumentPreprocessor,
        )
        from src.models.inference import (
            EnsemblePromptBuilder,
            ProFamEnsembleSampler,
            ProFamSampler,
            PromptBuilder,
        )
        from src.sequence.fasta import output_fasta

        # Build a ProteinDocument from the prompt sequences
        rep = prompt_names[0] if prompt_names else "representative"
        pool = ProteinDocument(
            sequences=prompt_seqs,
            accessions=prompt_names,
            identifier="profam_input",
            representative_accession=rep,
        )

        # Compute generation length cap
        longest_prompt_len = int(max(len(s) for s in prompt_seqs))
        max_sequence_length_multiplier = 1.2
        default_cap = int(longest_prompt_len * max_sequence_length_multiplier)
        if self.cfg.profam_max_generated_length is None:
            max_gen_len = default_cap
        else:
            max_gen_len = min(int(self.cfg.profam_max_generated_length), default_cap)

        doc_token = "[RAW]"

        # Build preprocessor and sampler
        if self.cfg.profam_sampler == "ensemble":
            preproc_cfg = AlignedProteinPreprocessingConfig(
                document_token=doc_token,
                defer_sampling=True,
                padding="do_not_pad",
                shuffle_proteins_in_document=True,
                keep_insertions=True,
                to_upper=True,
                keep_gaps=False,
                use_msa_pos=False,
                max_tokens_per_example=None,
            )
            preprocessor = ProteinDocumentPreprocessor(cfg=preproc_cfg)
            builder = EnsemblePromptBuilder(
                preprocessor=preprocessor, shuffle=True, seed=self.cfg.random_seed,
            )
            sampler_obj = ProFamEnsembleSampler(
                name="ensemble_sampler",
                model=self.model,
                prompt_builder=builder,
                document_token=doc_token,
                reduction="mean_probs",
                temperature=self.cfg.profam_temperature,
                top_p=self.cfg.profam_top_p,
                add_final_sep=True,
            )
            sampler_obj.to(self.device)
            sequences, scores, _ = sampler_obj.sample_seqs_ensemble(
                protein_document=pool,
                num_samples=num_samples,
                max_tokens=self.cfg.profam_max_tokens,
                num_prompts_in_ensemble=min(8, len(pool.sequences)),
                max_generated_length=max_gen_len,
                continuous_sampling=False,
                minimum_sequence_length_proportion=0.5,
                minimum_sequence_identity=None,
                maximum_retries=5,
                repeat_guard=True,
            )
        else:
            preproc_cfg = AlignedProteinPreprocessingConfig(
                document_token=doc_token,
                defer_sampling=False,
                padding="do_not_pad",
                shuffle_proteins_in_document=True,
                keep_insertions=True,
                to_upper=True,
                keep_gaps=False,
                use_msa_pos=False,
                max_tokens_per_example=self.cfg.profam_max_tokens - max_gen_len,
            )
            preprocessor = ProteinDocumentPreprocessor(cfg=preproc_cfg)
            builder = PromptBuilder(
                preprocessor=preprocessor, prompt_is_aligned=True, seed=self.cfg.random_seed,
            )
            sampling_kwargs: Dict[str, Any] = {}
            if self.cfg.profam_top_p is not None:
                sampling_kwargs["top_p"] = self.cfg.profam_top_p
            if self.cfg.profam_temperature is not None:
                sampling_kwargs["temperature"] = self.cfg.profam_temperature
            gen_batch_size = self.cfg.profam_generation_batch_size or num_samples
            sampling_kwargs["batch_generation"] = True
            sampling_kwargs["generation_batch_size"] = gen_batch_size
            sampler_obj = ProFamSampler(
                name="single_sampler",
                model=self.model,
                prompt_builder=builder,
                document_token=doc_token,
                sampling_kwargs=sampling_kwargs if sampling_kwargs else None,
                add_final_sep=True,
            )
            sampler_obj.to(self.device)
            sample_kwargs: Dict[str, Any] = dict(
                protein_document=pool,
                num_samples=num_samples,
                max_tokens=self.cfg.profam_max_tokens,
                max_generated_length=max_gen_len,
                continuous_sampling=False,
                minimum_sequence_length_proportion=0.5,
                minimum_sequence_identity=None,
                maximum_retries=5,
                repeat_guard=True,
            )
            sequences, scores, _ = sampler_obj.sample_seqs(**sample_kwargs)

        # Build accession names
        base = "profam_input"
        accessions = [
            f"{base}_sample_{i}_log_likelihood_{score:.3f}"
            for i, score in enumerate(scores)
        ]

        # Optionally save generated FASTA for debugging
        profam_out_dir = cycle_dir / "profam_outputs"
        profam_out_dir.mkdir(parents=True, exist_ok=True)
        out_fasta = profam_out_dir / f"{base}_generated_{self.cfg.profam_sampler}.fasta"
        output_fasta(accessions, sequences, str(out_fasta))

        return list(accessions), list(sequences)

    @property
    def method_name(self) -> str:
        return "profam"


class RandomMutationProposalGenerator(ProposalGenerator):
    """Generates candidate sequences by random point mutations."""

    def __init__(
        self,
        max_mutations: int,
        rng: np.random.Generator,
    ):
        """Initialize the random mutation generator.

        Args:
            max_mutations: Maximum number of mutations per candidate.
            rng: Random number generator for reproducibility.
        """
        self.max_mutations = max_mutations
        self.rng = rng

    def generate(
        self,
        prompt_seqs: List[str],
        prompt_names: List[str],
        num_samples: int,
        cycle_dir: Path,
    ) -> Tuple[List[str], List[str]]:
        aa_list = list(STANDARD_AMINO_ACIDS)
        names: List[str] = []
        sequences: List[str] = []

        for i in range(num_samples):
            parent = prompt_seqs[self.rng.integers(len(prompt_seqs))]
            seq = list(parent)
            n_mut = int(self.rng.integers(1, self.max_mutations + 1))
            n_mut = min(n_mut, len(seq))
            positions = self.rng.choice(len(seq), size=n_mut, replace=False)

            mutations_desc: List[str] = []
            for pos in positions:
                old_aa = seq[pos]
                new_aa = aa_list[self.rng.integers(len(aa_list))]
                mutations_desc.append(f"{old_aa}{pos + 1}{new_aa}")
                seq[pos] = new_aa

            mutant = "".join(seq)
            names.append(f"random_mutant_{i}_{'+'.join(mutations_desc)}")
            sequences.append(mutant)

        return names, sequences

    @property
    def method_name(self) -> str:
        return "random_mutation"
