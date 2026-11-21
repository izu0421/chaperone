"""Parse an AF3 candidate list into internal records.

A candidate is a receptor and a partner. The partner is either another
protein or a small molecule, and the two are triaged differently, so the kind
is resolved here rather than guessed downstream.

Column conventions accepted (case-insensitive):

  protein_a/protein_b/iptm/ptm/pae_interaction/model_path   the generic schema
  ligand/receptor/af3_iptm/af3_ptm                          a ligand-receptor
      PROTEIN screen; `ligand` here names a protein (a secreted ligand), not
      a small molecule, and is kept as protein_a for backward compatibility
  receptor + metabolite/ligand_smiles/ligand_ccd            a protein-ligand
      candidate: the small molecule is whichever of those columns is filled

A row is read as protein-ligand only when it carries a small-molecule column
with a value — `partner_kind`/`kind` ("protein"/"ligand"/"metabolite") sets
it explicitly when a file mixes both. The default stays protein-protein, so
an existing file parses exactly as it did before.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SMALL_MOLECULE_NAME_COLS = ("metabolite", "ligand_name", "small_molecule", "compound")
SMILES_COLS = ("ligand_smiles", "smiles", "partner_smiles")
CCD_COLS = ("ligand_ccd", "ccd", "ccd_code", "ligand_ccd_code")
KIND_COLS = ("partner_kind", "kind", "partner_type")
LIGAND_KINDS = {"ligand", "metabolite", "small_molecule", "smallmolecule", "compound", "drug"}


@dataclass
class Candidate:
    protein_a: str
    protein_b: str
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    pae_interaction: Optional[float] = None
    model_path: Optional[str] = None
    extra: dict = field(default_factory=dict)
    # protein-ligand candidates only
    partner_kind: str = "protein"
    ligand_name: Optional[str] = None
    ligand_smiles: Optional[str] = None
    ligand_ccd: Optional[str] = None

    @property
    def is_ligand(self) -> bool:
        return self.partner_kind == "ligand"

    @property
    def partner_label(self) -> str:
        """What to call the partner in a prompt or a report."""
        if not self.is_ligand:
            return self.protein_b
        return self.ligand_name or self.ligand_ccd or self.ligand_smiles or self.protein_b

    @property
    def pair_id(self) -> str:
        partner = self.partner_label if self.is_ligand else self.protein_b
        return f"{self.protein_a}__{partner}"


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _first(row: dict, names) -> Optional[str]:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return None


def load_candidates(csv_path: str) -> list[Candidate]:
    known_cols = {"protein_a", "protein_b", "iptm", "ptm", "pae_interaction", "model_path"}
    candidates = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        normalized_fields = {name: name.strip().lower() for name in reader.fieldnames or []}
        available = set(normalized_fields.values())
        iptm_col = "af3_iptm" if "af3_iptm" in available and "iptm" not in available else "iptm"
        ptm_col = "af3_ptm" if "af3_ptm" in available and "ptm" not in available else "ptm"
        # The protein side: `receptor` is the protein of a small-molecule row,
        # and the second protein of a ligand-receptor PROTEIN row, so which
        # column holds it is decided per row rather than per file — a file may
        # carry both kinds.
        protein_cols = [c for c in ("protein_a", "ligand", "receptor", "gene", "target") if c in available]
        partner_cols = [c for c in ("protein_b", "partner", "receptor", "partner_gene") if c in available]

        for raw in reader:
            row = {normalized_fields[k]: v for k, v in raw.items() if k in normalized_fields}
            declared = (_first(row, KIND_COLS) or "").lower()
            name = _first(row, SMALL_MOLECULE_NAME_COLS)
            smiles = _first(row, SMILES_COLS)
            ccd = _first(row, CCD_COLS)
            is_ligand = declared in LIGAND_KINDS or bool(name or smiles or ccd)
            if declared == "protein":
                is_ligand = False

            if is_ligand:
                # one protein chain: prefer the column that names a receptor
                protein_a = _first(row, ("protein_a", "receptor", "gene", "target", "ligand")) or ""
                protein_b = ""
                if not name:
                    # no name column, so the partner column holds the molecule
                    name = _first(row, [c for c in partner_cols if c != "receptor"])
            else:
                protein_a = _first(row, protein_cols) or ""
                protein_b = _first(row, [c for c in partner_cols if c != protein_cols[:1][0]]
                                   if protein_cols else partner_cols) or ""

            used = known_cols | {iptm_col, ptm_col} | set(protein_cols) | set(partner_cols)
            used |= set(SMALL_MOLECULE_NAME_COLS + SMILES_COLS + CCD_COLS + KIND_COLS)
            candidates.append(
                Candidate(
                    protein_a=protein_a,
                    protein_b=protein_b,
                    iptm=_to_float(row.get(iptm_col)),
                    ptm=_to_float(row.get(ptm_col)),
                    pae_interaction=_to_float(row.get("pae_interaction")),
                    model_path=row.get("model_path") or None,
                    partner_kind="ligand" if is_ligand else "protein",
                    ligand_name=name,
                    ligand_smiles=smiles,
                    ligand_ccd=ccd,
                    extra={k: v for k, v in row.items() if k not in used},
                )
            )
    return candidates


if __name__ == "__main__":
    import sys

    for c in load_candidates(sys.argv[1]):
        print(c)
