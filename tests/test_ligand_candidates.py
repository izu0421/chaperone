"""Protein-ligand candidates: parsing, fold input, and the gate's rules.

The gate rules are the part worth testing hardest. A small molecule folds
confidently into almost any pocket, so the rules that flag a confident but
non-specific result are what stop this pipeline reporting a pose as a
finding.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chaperone.deterministic_gate import compute_gate_facts, find_contradictions  # noqa: E402
from chaperone.fold_candidate import build_af3_input  # noqa: E402
from chaperone.ingest import load_candidates  # noqa: E402


def _csv(tmp_path, rows, header):
    path = tmp_path / "candidates.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


# --------------------------------------------------------------- ingest ----

def test_metabolite_column_makes_a_ligand_candidate(tmp_path):
    path = _csv(tmp_path, [["GPR35", "kynurenic acid", "0.86"]],
                ["receptor", "metabolite", "af3_iptm"])
    c, = load_candidates(path)
    assert c.is_ligand and c.partner_kind == "ligand"
    assert c.protein_a == "GPR35" and c.ligand_name == "kynurenic acid"
    assert c.partner_label == "kynurenic acid"
    assert c.pair_id == "GPR35__kynurenic acid"
    assert c.iptm == 0.86


def test_ligand_receptor_protein_schema_is_unchanged(tmp_path):
    """`ligand` in the existing screen names a PROTEIN, not a small molecule
    — reading it as one would silently retype every row of an existing file."""
    path = _csv(tmp_path, [["IL34", "CSF1R", "0.72"]], ["ligand", "receptor", "af3_iptm"])
    c, = load_candidates(path)
    assert not c.is_ligand
    assert (c.protein_a, c.protein_b) == ("IL34", "CSF1R")


def test_partner_kind_column_mixes_both_in_one_file(tmp_path):
    path = _csv(
        tmp_path,
        [["GPR35", "kynurenate", "ligand", "C1=CC=CC=C1", "0.86"],
         ["GPR35", "CXCL17", "protein", "", "0.55"]],
        ["receptor", "partner", "partner_kind", "ligand_smiles", "af3_iptm"],
    )
    lig, prot = load_candidates(path)
    assert lig.is_ligand and lig.ligand_smiles == "C1=CC=CC=C1"
    assert lig.partner_label == "kynurenate"
    assert not prot.is_ligand and prot.protein_b == "CXCL17"


def test_smiles_only_row_still_names_the_partner(tmp_path):
    path = _csv(tmp_path, [["GPR35", "CCO"]], ["receptor", "ligand_smiles"])
    c, = load_candidates(path)
    assert c.is_ligand and c.ligand_smiles == "CCO"
    assert c.partner_label == "CCO"        # no name given, so the structure stands in


# ----------------------------------------------------------- fold input ----

def test_ligand_becomes_its_own_af3_chain():
    chains = [{"sequence": "MKV"}]
    spec = build_af3_input("j", chains, [1], ligands=[{"smiles": "CCO", "name": "ethanol"}])
    assert spec["sequences"][0]["protein"]["id"] == ["A"]
    assert spec["sequences"][1]["ligand"] == {"id": ["B"], "smiles": "CCO"}


def test_ccd_code_is_preferred_over_smiles_when_given():
    spec = build_af3_input("j", [{"sequence": "MKV"}], [1],
                           ligands=[{"ccd": "ATP", "smiles": "ignored"}])
    assert spec["sequences"][1]["ligand"] == {"id": ["B"], "ccdCodes": ["ATP"]}


def test_ligand_without_a_structure_is_refused():
    """An unresolved molecule name must not fold a protein on its own and
    report the result as a complex."""
    try:
        build_af3_input("j", [{"sequence": "MKV"}], [1], ligands=[{"name": "mystery"}])
    except ValueError as exc:
        assert "ccd" in str(exc) and "smiles" in str(exc)
    else:
        raise AssertionError("a ligand with no structure should be refused")


def test_glycan_chains_start_after_the_ligand():
    spec = build_af3_input("j", [{"sequence": "MKV"}], [1],
                           glycan_sequences=[{"ligand": {"id": ["C"], "ccdCodes": ["NAG"]}}],
                           ligands=[{"ccd": "ATP"}])
    ids = [list(e.values())[0]["id"][0] for e in spec["sequences"]]
    assert ids == ["A", "B", "C"]          # no two chains share a letter


# ------------------------------------------------------------- the gate ----

def _ligand_row(verdict):
    return {"protein_a": "GPR35", "protein_b": "serotonin",
            "partner_kind": "ligand", "verdict": verdict}


def _calls(n_activities=None, n_targets=None):
    calls = []
    if n_activities is not None:
        calls.append({"tool": "chembl_known_activity",
                      "output": {"n_activities": n_activities}})
    if n_targets is not None:
        calls.append({"tool": "chembl_ligand_promiscuity",
                      "output": {"n_distinct_targets_sampled": n_targets}})
    return calls


def _flags(verdict, calls):
    row = _ligand_row(verdict)
    return find_contradictions(row, compute_gate_facts(row, calls, []))


def test_measured_affinity_at_this_receptor_contradicts_novelty():
    flags = _flags("CONFIRMED_NOVEL", _calls(n_activities=116, n_targets=3))
    assert any("ALREADY_KNOWN" in f for f in flags)


def test_measured_affinity_does_not_fire_when_already_known():
    assert _flags("ALREADY_KNOWN", _calls(n_activities=116, n_targets=3)) == []


def test_promiscuous_molecule_contradicts_confirmed_novel():
    flags = _flags("CONFIRMED_NOVEL", _calls(n_activities=0, n_targets=79))
    assert any("LIKELY_PROMISCUOUS_POCKET" in f for f in flags)


def test_promiscuity_does_not_fire_on_a_selective_molecule():
    assert _flags("CONFIRMED_NOVEL", _calls(n_activities=0, n_targets=2)) == []


def test_unchecked_small_molecule_evidence_is_flagged():
    flags = _flags("CONFIRMED_NOVEL", [])
    assert any("chembl_known_activity" in f for f in flags)


def test_protein_rows_are_untouched_by_the_ligand_rules():
    """compute_gate_facts must not run the small-molecule rules on a protein
    pair, nor the two-gene HPA/glycan checks on a molecule name."""
    row = {"protein_a": "IL34", "protein_b": "CSF1R", "verdict": "CONFIRMED_NOVEL"}
    facts = compute_gate_facts(row, [], [])
    assert "ligand" not in facts and facts["partner_kind"] == "protein"
    ligand_facts = compute_gate_facts(_ligand_row("CONFIRMED_NOVEL"), [], [])
    assert ligand_facts["hpa_plausibility"] == "not_applicable"
    assert ligand_facts["glycan_binding_gene"] is None


def test_unusable_promiscuity_count_is_not_read_as_selective():
    """ChEMBL returned activities with no target id on any of them, making the
    distinct-target count unavailable. Reading that as 0 would mark a
    promiscuous molecule selective — the one direction that waves a
    non-specific hit through."""
    calls = [{"tool": "chembl_known_activity", "output": {"n_activities": 0}},
             {"tool": "chembl_ligand_promiscuity",
              "output": {"error": "no target id on any of them",
                         "n_affinity_measurements": 638}}]
    row = _ligand_row("CONFIRMED_NOVEL")
    facts = compute_gate_facts(row, calls, [])
    assert facts["ligand"]["n_distinct_targets"] is None
    assert facts["ligand"]["promiscuous"] is False     # unknown, not selective
    assert facts["ligand"]["checked"] is False
    assert any("chembl_known_activity" in f or "promiscuity" in f
               for f in find_contradictions(row, facts))


# ------------------------------------------------- ligand pose geometry ----

def test_sasa_matches_the_analytic_sphere():
    """The Shrake-Rupley implementation is written here rather than imported,
    so it needs its own check against a case with a closed form."""
    import numpy as np
    from chaperone.analyze_fold import _sasa, PROBE
    area = _sasa(np.array([[0.0, 0.0, 0.0]]), np.array([1.70]))[0]
    assert abs(area - 4 * np.pi * (1.70 + PROBE) ** 2) / area < 0.01


def test_two_coincident_atoms_do_not_double_count_surface():
    import numpy as np
    from chaperone.analyze_fold import _sasa
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])
    one = _sasa(np.array([[0.0, 0.0, 0.0]]), np.array([1.70]))[0]
    assert abs(_sasa(coords, np.array([1.70, 1.70])).sum() - one) / one < 0.05


def test_ligand_geometry_reports_burial_clash_and_pocket():
    """A molecule enclosed by a shell of protein atoms should read as buried,
    unclashed, and in contact with the residues that enclose it."""
    import numpy as np
    from chaperone.analyze_fold import ligand_geometry
    # a shell at 4.4 A: close enough that each atom's probe-expanded sphere
    # (1.70 + 1.4) reaches past the ligand's own, which is what burial means.
    # A sparser cage at 4 A on the diagonal sits 6.9 A out and occludes
    # nothing, which is the right answer for a cage, not a pocket.
    directions = []
    for ax in range(3):
        for sign in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0]
            v[ax] = sign
            directions.append(v)
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                n = 3 ** 0.5
                directions.append([x / n, y / n, z / n])
    shell = np.array(directions) * 4.4
    chains = {
        "A": {"coords": shell, "radii": np.full(len(shell), 1.70),
              "res_index": np.arange(1, len(shell) + 1), "n_atoms": len(shell),
              "plddt": np.full(len(shell), 80.0), "is_ligand": False},
        "B": {"coords": np.array([[0.0, 0.0, 0.0]]), "radii": np.array([1.70]),
              "res_index": np.array([1]), "n_atoms": 1,
              "plddt": np.array([72.0]), "is_ligand": True},
    }
    g = ligand_geometry(chains, "B", {"A": []})
    assert g["ligand_chain"] == "B" and g["n_ligand_atoms"] == 1
    assert g["ligand_plddt_mean"] == 72.0
    assert g["buried_fraction"] > 0.5          # enclosed by the shell
    assert g["clash"] is False                 # nearest shell atom is ~6.9 A
    assert g["pocket"]["A"]["n_contacting_residues"] > 0


def test_ligand_geometry_flags_a_clash():
    import numpy as np
    from chaperone.analyze_fold import ligand_geometry
    chains = {
        "A": {"coords": np.array([[1.0, 0.0, 0.0]]), "radii": np.array([1.70]),
              "res_index": np.array([1]), "n_atoms": 1,
              "plddt": np.array([80.0]), "is_ligand": False},
        "B": {"coords": np.array([[0.0, 0.0, 0.0]]), "radii": np.array([1.70]),
              "res_index": np.array([1]), "n_atoms": 1,
              "plddt": np.array([60.0]), "is_ligand": True},
    }
    g = ligand_geometry(chains, "B", {"A": []})
    assert g["min_separation_A"] == 1.0 and g["clash"] is True
