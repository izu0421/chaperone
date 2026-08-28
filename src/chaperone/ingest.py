"""Parse an AF3 novel-PPI candidate list into internal records.

Two column naming conventions are accepted (case-insensitive):
- protein_a/protein_b/iptm/ptm/pae_interaction/model_path — the generic schema
- ligand/receptor/af3_iptm/af3_ptm — the real schema used in
  data/*promising_receptor_ligand_pairs*.csv (a ligand-receptor AF2+AF3
  screen), which also carries af2_iptm/af2_ptm/n_samples_colocalized/
  prop_samples_colocalized/*_uniprot/*_sequence columns through as `extra`
  (prop_samples_colocalized in particular is real observed spatial
  co-localization evidence — stronger than anything HPA/STRING infer, so the
  agent is given it directly when present).
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Candidate:
    protein_a: str
    protein_b: str
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    pae_interaction: Optional[float] = None
    model_path: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def pair_id(self) -> str:
        return f"{self.protein_a}__{self.protein_b}"


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_candidates(csv_path: str) -> list[Candidate]:
    known_cols = {"protein_a", "protein_b", "iptm", "ptm", "pae_interaction", "model_path"}
    candidates = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        normalized_fields = {name: name.strip().lower() for name in reader.fieldnames or []}
        available = set(normalized_fields.values())
        protein_a_col = "ligand" if "ligand" in available and "protein_a" not in available else "protein_a"
        protein_b_col = "receptor" if "receptor" in available and "protein_b" not in available else "protein_b"
        iptm_col = "af3_iptm" if "af3_iptm" in available and "iptm" not in available else "iptm"
        ptm_col = "af3_ptm" if "af3_ptm" in available and "ptm" not in available else "ptm"

        for row in reader:
            row = {normalized_fields[k]: v for k, v in row.items() if k in normalized_fields}
            candidates.append(
                Candidate(
                    protein_a=row[protein_a_col].strip(),
                    protein_b=row[protein_b_col].strip(),
                    iptm=_to_float(row.get(iptm_col)),
                    ptm=_to_float(row.get(ptm_col)),
                    pae_interaction=_to_float(row.get("pae_interaction")),
                    model_path=row.get("model_path") or None,
                    extra={k: v for k, v in row.items() if k not in known_cols | {protein_a_col, protein_b_col, iptm_col, ptm_col}},
                )
            )
    return candidates


if __name__ == "__main__":
    import sys

    for c in load_candidates(sys.argv[1]):
        print(c)
