"""Pipeline sub-package for the ProFam + BAGEL generative design pipeline."""

from pipeline.colabfold_oracle import ColabFold, ColabFoldResult
from pipeline.grpo import GRPOConfig, PipelineGRPOStep
from pipeline.bandits import (
    ProposalBandit,
    TemperatureBandit,
    ThompsonArm,
    ThompsonSampler,
)
from pipeline.logging import (
    append_cycle_csv,
    save_selected_structures,
    update_cycle_log,
)
from pipeline.plotting import make_energy_summary_plot
from pipeline.proposal import (
    ProFamProposalGenerator,
    ProposalGenerator,
    RandomMutationProposalGenerator,
)
from pipeline.selection import (
    GreedyPromptSelector,
    PromptSelector,
    SelectionManager,
    SelectionResult,
    ThompsonPromptSelector,
)
from pipeline.utils import (
    compute_avg_sequence_similarity,
    compute_sequence_identity,
    extract_reward_term,
    sample_subset_indices,
    softmax_from_energies,
)

__all__ = [
    "ColabFold",
    "ColabFoldResult",
    "GRPOConfig",
    "PipelineGRPOStep",
    "ThompsonArm",
    "ThompsonSampler",
    "TemperatureBandit",
    "ProposalBandit",
    "compute_sequence_identity",
    "compute_avg_sequence_similarity",
    "extract_reward_term",
    "softmax_from_energies",
    "sample_subset_indices",
    "update_cycle_log",
    "append_cycle_csv",
    "save_selected_structures",
    "make_energy_summary_plot",
    "ProposalGenerator",
    "ProFamProposalGenerator",
    "RandomMutationProposalGenerator",
    "SelectionResult",
    "PromptSelector",
    "GreedyPromptSelector",
    "ThompsonPromptSelector",
    "SelectionManager",
]
