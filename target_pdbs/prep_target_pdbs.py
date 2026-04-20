#!/usr/bin/env python3
"""Extract single chain from each target PDB for BindCraft-style AF2 predictions."""
from Bio.PDB import PDBParser, PDBIO, Select

# Map target_id -> (pdb_file, chain_id) for the chain matching the target sequence
TARGET_CHAINS = {
    "2GDZ": ("2GDZ.pdb", "A"),
    "1YCR_MDM2": ("1YCR.pdb", "A"),
    "4OYD_epstein_barr": ("4OYD.pdb", "A"),
    "4ZQK_PD-L1": ("4ZQK.pdb", "A"),
    "1TNF_TNF_alpha": ("1TNF.pdb", "A"),
    "2VSM_nipah": ("2VSM.pdb", "A"),
}


class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        # Standard residues only (no HETATM water/ligands)
        return residue.id[0] == " "


def main():
    parser = PDBParser(QUIET=True)
    io = PDBIO()
    for tid, (pdb_file, chain_id) in TARGET_CHAINS.items():
        s = parser.get_structure(tid, pdb_file)
        out = f"{tid}.pdb"
        io.set_structure(s)
        io.save(out, ChainSelect(chain_id))
        # Count residues in the saved file
        s2 = parser.get_structure(tid, out)
        n = sum(1 for r in s2.get_residues() if r.id[0] == " ")
        print(f"  {tid}: chain {chain_id} → {out} ({n} residues)")


if __name__ == "__main__":
    main()
