"""Custom energy terms for the ProFam + BAGEL pipeline.

These extend BAGEL's built-in energies and are auto-registered into
``bagel.energies`` so the YAML config loader can find them via
``getattr(bg.energies, etype)``.
"""

from __future__ import annotations

import bagel.energies as _bg_energies
from bagel.energies import EnergyTerm
from bagel.oracles.base import Oracle, OraclesResultDict


class OneSidedBinderLengthEnergy(EnergyTerm):
    """Penalize binder length only when it exceeds a threshold.

    Counts residues only in the GEN chain (the binder), ignoring the target.
    Energy is zero below the threshold and rises linearly above it:

        value = penalty_per_residue * max(0, num_residues - threshold)

    This is a one-sided alternative to BAGEL's ChemicalPotentialEnergy
    (which uses |num_residues - target_size| and penalizes both directions).
    """

    def __init__(
        self,
        oracle: Oracle,
        threshold: int = 120,
        penalty_per_residue: float = 0.01,
        binder_chain_id: str = "GEN",
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = "binder_length"
        else:
            name = f"binder_length_{name}"
        super().__init__(name=name, inheritable=True, oracle=oracle, weight=weight)
        self.threshold = threshold
        self.penalty_per_residue = penalty_per_residue
        self.binder_chain_id = binder_chain_id
        assert isinstance(self.oracle, Oracle), "Input to oracle not an Oracle object"
        assert "input_chains" in self.oracle.result_class.model_fields, (
            "OneSidedBinderLengthEnergy requires oracle to return input_chains"
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        input_chains = oracles_result.get_input_chains(self.oracle)
        binder_length = 0
        for chain in input_chains:
            if chain.chain_ID == self.binder_chain_id:
                binder_length += chain.length
        excess = max(0, binder_length - self.threshold)
        value = self.penalty_per_residue * excess
        return value, value * self.weight


# Register custom energies into bagel.energies so the YAML loader picks them up
_bg_energies.OneSidedBinderLengthEnergy = OneSidedBinderLengthEnergy
