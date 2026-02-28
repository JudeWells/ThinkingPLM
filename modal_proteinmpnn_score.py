"""
Standalone Modal app for scoring protein sequences with SolubleMPNN.

Runs in an isolated container with numpy 1.x / torch 2.2.1 to avoid
dependency conflicts with the main BAGEL + Boltz pipeline.

Deploy once:
    modal deploy modal_proteinmpnn_score.py

Call from other Modal containers:
    import modal
    score_fn = modal.Function.from_name("proteinmpnn-scorer", "score_sequence")
    result = score_fn.remote(pdb_str, chains_to_score="A")
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install([
        "numpy==1.23.5",
        "torch==2.2.1",
    ])
    .run_commands(
        "git clone https://github.com/dauparas/ProteinMPNN.git /ProteinMPNN"
    )
)

app = modal.App("proteinmpnn-scorer", image=image)


@app.function(gpu="T4", timeout=300)
def score_sequence(
    pdb_str: str,
    chains_to_score: str = "A",
    num_batches: int = 10,
) -> dict:
    """
    Score a sequence's perplexity given its backbone structure using SolubleMPNN.

    The full complex (all chains) is passed to the encoder so the model sees
    the binding context, but only residues on ``chains_to_score`` contribute
    to the perplexity.

    Parameters
    ----------
    pdb_str : str
        PDB file contents as a string (full complex, all chains).
    chains_to_score : str
        Comma-separated chain IDs to compute perplexity for (e.g. "A" or "A,C").
    num_batches : int
        Number of random decoding orders to average over.

    Returns
    -------
    dict with keys:
        perplexity : float
            exp(mean NLL) over scored chains (lower = better designability).
        mean_nll : float
            Mean negative log-likelihood per residue on scored chains.
        std_nll : float
            Std of NLL across decoding orders.
        global_perplexity : float
            Perplexity computed over ALL chains (for reference).
    """
    import sys
    sys.path.insert(0, "/ProteinMPNN")

    import copy
    import tempfile
    import os

    import numpy as np
    import torch
    from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize, _scores

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load SolubleMPNN model ---
    checkpoint_path = "/ProteinMPNN/soluble_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    hidden_dim = 128
    num_layers = 3
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Write PDB string to a temp file and parse ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False
    ) as fh:
        fh.write(pdb_str)
        tmp_pdb = fh.name

    try:
        pdb_dict_list = parse_PDB(tmp_pdb, ca_only=False)
    finally:
        os.unlink(tmp_pdb)

    if not pdb_dict_list:
        raise ValueError("parse_PDB returned empty list — check PDB string.")

    pdb_dict = pdb_dict_list[0]

    # --- Determine designed vs fixed chains ---
    designed = [c.strip() for c in chains_to_score.split(",")]
    all_chains = sorted(
        k.split("_")[-1] for k in pdb_dict if k.startswith("seq_chain_")
    )
    fixed = [c for c in all_chains if c not in designed]

    chain_id_dict = {pdb_dict["name"]: (designed, fixed)}

    # --- Featurise ---
    batch = [copy.deepcopy(pdb_dict)]
    (
        X, S, mask, lengths, chain_M, chain_encoding_all,
        chain_list_list, visible_list_list, masked_list_list,
        masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
        residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
        pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all,
        tied_beta,
    ) = tied_featurize(
        batch, device, chain_id_dict, ca_only=False,
    )

    # --- Score with multiple decoding orders ---
    score_list = []
    global_score_list = []

    with torch.no_grad():
        for _ in range(num_batches):
            randn = torch.randn(chain_M.shape, device=device)
            log_probs = model(
                X, S, mask,
                chain_M * chain_M_pos,
                residue_idx, chain_encoding_all,
                randn,
            )
            mask_for_loss = mask * chain_M * chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            global_scores = _scores(S, log_probs, mask)
            score_list.append(scores.cpu().numpy())
            global_score_list.append(global_scores.cpu().numpy())

    all_scores = np.concatenate(score_list, 0)
    all_global = np.concatenate(global_score_list, 0)

    mean_nll = float(all_scores.mean())
    return {
        "perplexity": float(np.exp(mean_nll)),
        "mean_nll": mean_nll,
        "std_nll": float(all_scores.std()),
        "global_perplexity": float(np.exp(all_global.mean())),
    }
