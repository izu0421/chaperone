"""Attach real UniProt-annotated N-glycosylation sites as covalently-bonded
sugar chains in an AF3 input, instead of folding a bare, unmodified sequence.
This directly closes the gap behind every `LIKELY_ARTIFACT_PTM` verdict that
cites "AF3's bare-sequence model can't represent glycans": AlphaFold 3 (the
real underlying model in the local AlphaFast install, confirmed via its own
docs/input_format.md and folding_input.py) supports ligand chains built from
CCD sugar codes plus explicit `bondedAtomPairs` covalent bonds — it was just
never used by fold_candidate.py, which only ever emitted bare protein chains.
Each N-glycosylation site gets the minimal, standard N-glycan core
trisaccharide: Asn-GlcNAc(NAG)-GlcNAc(NAG)-Man(BMA), the conserved inner
core shared by every N-glycan regardless of its outer branching (which
UniProt's sequence-level annotation can't tell you anyway — see
RESULTS/discussion: no PDB structure resolves ADGRL2's actual glycan, so a
generic-but-real core is the honest, defensible choice, not a guess at the
full branched tree).
CCD atom names (NAG.cif / ASN.cif from the RCSB Chemical Component
Dictionary, fetched directly rather than assumed):
  - NAG: anomeric carbon C1, O4 is the position the NEXT sugar's C1 attaches
    to for a beta-1,4 linkage (the real chitobiose/core-mannose linkage).
  - ASN: side-chain amide nitrogen is ND2 - the real N-glycosidic bond is
    innermost-NAG:C1 <-> ASN:ND2.
"""
import httpx
from .sources.http_retry import get_with_retry
# Minimal, standard N-glycan core: chitobiose (2x GlcNAc) + the first core
# mannose. Every real N-glycan (high-mannose, hybrid, or complex-type) shares
# exactly this inner core; only the outer branching (which UniProt sequence
# annotation cannot tell us) differs.
CORE_N_GLYCAN_CCD = ["NAG", "NAG", "BMA"]
def fetch_n_glycosylation_sites(accession: str) -> list[int]:
    """Real UniProt 'Glycosylation' features for this accession, restricted
    to N-linked sites (the type AF3's CCD-ligand system can represent as a
    single-residue attachment point) and single-residue annotations (a
    glycosylation feature always is - start==end for a covalent-attachment
    PTM), returning the Asn residue position for each."""
    with httpx.Client() as client:
        resp = get_with_retry(client, f"https://rest.uniprot.org/uniprotkb/{accession}.json", {}, 30.0)
    features = resp.json().get("features", [])
    sites = []
    for f in features:
        if f["type"] != "Glycosylation":
            continue
        desc = (f.get("description") or "")
        if "N-linked" not in desc:
            continue
        pos = f["location"]["start"]["value"]
        sites.append(pos)
    return sorted(set(sites))
def build_glycan_chains_and_bonds(
    protein_chain_id: str, sites: list[int], chain_letters: list[str]
) -> tuple[list[dict], list]:
    """One 3-residue N-glycan-core ligand chain per site, using the given
    pool of unused chain letters (one per site; raises if not enough).
    Returns (af3 'ligand' sequence entries, bondedAtomPairs entries)."""
    if len(chain_letters) < len(sites):
        raise ValueError(f"Need {len(sites)} free chain letters for glycans, got {len(chain_letters)}")
    ligand_sequences, bonds = [], []
    for site, letter in zip(sites, chain_letters):
        ligand_sequences.append({"ligand": {"id": [letter], "ccdCodes": list(CORE_N_GLYCAN_CCD)}})
        # Asn:ND2 <-> innermost NAG (ligand residue 1):C1 - the real
        # N-glycosidic bond.
        bonds.append([[protein_chain_id, site, "ND2"], [letter, 1, "C1"]])
        # NAG1:O4 <-> NAG2:C1, NAG2:O4 <-> BMA:C1 - the real core linkages.
        bonds.append([[letter, 1, "O4"], [letter, 2, "C1"]])
        bonds.append([[letter, 2, "O4"], [letter, 3, "C1"]])
    return ligand_sequences, bonds
