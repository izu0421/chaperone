"""Tests for the deterministic gate's pure fact-computing functions — the
part of chaperone that must be reproducible (same evidence in, same fact
out, no LLM). Fixtures mirror the real tool-output JSON shapes confirmed
live against the actual HPA/STRING/CellPhoneDB/fold APIs during
development; see chaperone/sources/*_client.py and analyze_fold.py.
"""
import chaperone.deterministic_gate as gate


def _call(tool, output):
    return {"tool": tool, "output": output}


# ---------------------------------------------------------------- known_interaction_strength

def test_cellphonedb_hit_is_confirmed_on_its_own():
    tool_calls = [_call("cellphonedb_known_interaction", {"found": True, "matches": ["integrin_a2b1_complex"]})]
    result = gate.known_interaction_strength(tool_calls)
    assert result["strength"] == "confirmed"


def test_string_textmining_only_is_weak_not_confirmed():
    # STRING complaint this was built to fix: a co-mention signal alone must
    # never disqualify novelty.
    tool_calls = [_call("string_known_interaction", {
        "found": True, "combined_score": 0.55,
        "evidence": {"database": 0, "experiments": 0, "textmining": 0.516, "coexpression": 0.108},
    })]
    result = gate.known_interaction_strength(tool_calls)
    assert result["strength"] == "weak"


def test_string_strong_channel_with_literature_is_confirmed():
    tool_calls = [
        _call("string_known_interaction", {
            "found": True, "combined_score": 0.87,
            "evidence": {"database": 0.5, "experiments": 0.1, "textmining": 0.7, "coexpression": 0.1},
        }),
        _call("pubmed_cooccurrence", {"count": 4, "hits": []}),
    ]
    result = gate.known_interaction_strength(tool_calls)
    assert result["strength"] == "confirmed"


def test_string_strong_channel_without_literature_stays_novel():
    # the exact real bug found this session: a curated pathway-membership
    # score with zero literature trace for THIS pair is not proof of a known
    # interaction.
    tool_calls = [
        _call("string_known_interaction", {
            "found": True, "combined_score": 0.514,
            "evidence": {"database": 0, "experiments": 0.514, "textmining": 0, "coexpression": 0},
        }),
        _call("pubmed_cooccurrence", {"count": 0, "hits": []}),
    ]
    result = gate.known_interaction_strength(tool_calls)
    assert result["strength"] == "weak"


def test_no_hits_anywhere_is_none():
    tool_calls = [
        _call("string_known_interaction", {"found": False}),
        _call("cellphonedb_known_interaction", {"found": False, "matches": []}),
    ]
    assert gate.known_interaction_strength(tool_calls)["strength"] == "none"


# ---------------------------------------------------------------- hpa_plausibility

def _protein_class(gene, protein_class, secretome_location=None):
    return _call("hpa_protein_class", {"gene": gene, "protein_class": protein_class, "secretome_location": secretome_location})


def _expression(gene, tissues=None, cell_types=None):
    return _call("hpa_expression", {
        "gene": gene,
        "rna_tissue_specific_nTPM": {t: "1.0" for t in (tissues or [])},
        "rna_single_cell_type_specific_nCPM": {c: "1.0" for c in (cell_types or [])},
    })


def test_membrane_and_intracellular_tag_together_is_not_a_mismatch():
    # the receptor-cytoplasmic-tail bug: a protein tagged BOTH membrane and
    # intracellular (very common — a receptor's own tail earns the second
    # tag) must not be treated as having "no membrane presence."
    tool_calls = [
        _protein_class("NPR3", ["Predicted membrane proteins", "Predicted intracellular proteins"]),
        _protein_class("RAB8B", ["Predicted intracellular proteins"]),
        _expression("NPR3", tissues=["kidney"], cell_types=["podocytes"]),
        _expression("RAB8B", tissues=["kidney"], cell_types=["podocytes"]),
    ]
    result = gate.hpa_plausibility(tool_calls, "NPR3", "RAB8B")
    assert result["plausibility"] == "plausible"


def test_no_mismatch_but_no_shared_tissue_and_neither_secreted_is_implausible():
    # two compatible (non-mismatched) compartments still can't interact if
    # they're never actually coexpressed anywhere, and neither is secreted
    # (so there's no circulation-based route for them to meet either).
    tool_calls = [
        _protein_class("GENE_A", ["Predicted membrane proteins"]),
        _protein_class("GENE_B", ["Predicted membrane proteins"]),
        _expression("GENE_A", tissues=["brain"], cell_types=["neurons"]),
        _expression("GENE_B", tissues=["liver"], cell_types=["hepatocytes"]),
    ]
    result = gate.hpa_plausibility(tool_calls, "GENE_A", "GENE_B")
    assert result["plausibility"] == "implausible"


def test_no_mismatch_no_shared_tissue_but_secreted_ligand_is_still_plausible():
    # a secreted ligand circulates — it doesn't need to be expressed in the
    # same tissue as its receptor (paracrine/endocrine signalling). A blanket
    # same-tissue filter would wrongly kill real ligand-receptor pairs.
    tool_calls = [
        _protein_class("LIGAND", ["Predicted secreted proteins"], secretome_location="Secreted"),
        _protein_class("RECEPTOR", ["Predicted membrane proteins"]),
        _expression("LIGAND", tissues=["liver"], cell_types=["hepatocytes"]),
        _expression("RECEPTOR", tissues=["brain"], cell_types=["neurons"]),
    ]
    result = gate.hpa_plausibility(tool_calls, "LIGAND", "RECEPTOR")
    assert result["plausibility"] == "plausible"


def test_no_mismatch_missing_expression_data_is_unknown_not_implausible():
    tool_calls = [
        _protein_class("GENE_A", ["Predicted membrane proteins"]),
        _protein_class("GENE_B", ["Predicted membrane proteins"]),
    ]
    result = gate.hpa_plausibility(tool_calls, "GENE_A", "GENE_B")
    assert result["plausibility"] == "unknown"


def test_intracellular_only_plus_secreted_with_no_coexpression_is_implausible():
    tool_calls = [
        _protein_class("GENE_A", ["Predicted intracellular proteins"]),
        _protein_class("GENE_B", ["Predicted secreted proteins"], secretome_location="Secreted"),
        _expression("GENE_A", tissues=["brain"], cell_types=["neurons"]),
        _expression("GENE_B", tissues=["liver"], cell_types=["hepatocytes"]),
    ]
    result = gate.hpa_plausibility(tool_calls, "GENE_A", "GENE_B")
    assert result["plausibility"] == "implausible"


def test_compartment_mismatch_with_real_coexpression_is_still_plausible():
    # proteins can moonlight/shuttle — a real shared tissue/cell type
    # overrides a naive compartment mismatch.
    tool_calls = [
        _protein_class("GENE_A", ["Predicted intracellular proteins"]),
        _protein_class("GENE_B", ["Predicted secreted proteins"], secretome_location="Secreted"),
        _expression("GENE_A", tissues=["bone marrow"], cell_types=["Neutrophils"]),
        _expression("GENE_B", tissues=["bone marrow"], cell_types=["Neutrophils"]),
    ]
    result = gate.hpa_plausibility(tool_calls, "GENE_A", "GENE_B")
    assert result["plausibility"] == "plausible"


def test_missing_protein_class_data_is_unknown_not_implausible():
    result = gate.hpa_plausibility([], "GENE_A", "GENE_B")
    assert result["plausibility"] == "unknown"


# ---------------------------------------------------------------- fold_evidence_status

def _interface(chain_a="A", chain_b="B", ptm_a=None, ptm_b=None, topo_a=None, topo_b=None):
    return {
        "chain_a": chain_a, "chain_b": chain_b,
        "ptm_sites_at_interface_a": ptm_a or [], "ptm_sites_at_interface_b": ptm_b or [],
        "interface_topology_a": topo_a or [], "interface_topology_b": topo_b or [],
    }


def test_no_fold_results_is_not_run():
    assert gate.fold_evidence_status([])["status"] == "not_run"


def test_clean_fold_is_pass():
    fold_results = [{"structural_analysis": {"interfaces": [_interface()]}}]
    assert gate.fold_evidence_status(fold_results)["status"] == "pass"


def test_disulfide_bond_is_not_an_artifact_risk():
    # the exact bug found this session: AF3 represents disulfide bonds
    # fine, unlike glycans — only Glycosylation/Lipidation should count.
    fold_results = [{"structural_analysis": {"interfaces": [
        _interface(ptm_a=[{"type": "Disulfide bond", "start": 10, "end": 20}])
    ]}}]
    assert gate.fold_evidence_status(fold_results)["status"] == "pass"


def test_glycosylation_at_interface_is_ptm_only_concern():
    fold_results = [{"structural_analysis": {"interfaces": [
        _interface(ptm_a=[{"type": "Glycosylation", "start": 10, "end": 10}])
    ]}}]
    assert gate.fold_evidence_status(fold_results)["status"] == "ptm_only_concern"


def test_membrane_topology_violation_outranks_ptm_concern():
    fold_results = [{"structural_analysis": {"interfaces": [
        _interface(
            ptm_a=[{"type": "Glycosylation", "start": 10, "end": 10}],
            topo_a=["Cytoplasmic"], topo_b=["Extracellular"],
        )
    ]}}]
    assert gate.fold_evidence_status(fold_results)["status"] == "topology_violation"


# ---------------------------------------------------------------- find_contradictions

def test_no_contradiction_when_verdict_matches_all_facts():
    row = {"verdict": "CONFIRMED_NOVEL"}
    facts = {
        "known_interaction_strength": "none", "known_interaction_trace": "",
        "hpa_plausibility": "plausible", "hpa_trace": "",
        "fold_evidence_status": "not_run", "fold_trace": "",
        "glycan_binding_gene": None,
    }
    assert gate.find_contradictions(row, facts) == []


def test_confirmed_known_interaction_but_verdict_says_otherwise_is_flagged():
    row = {"verdict": "CONFIRMED_NOVEL"}
    facts = {
        "known_interaction_strength": "confirmed", "known_interaction_trace": "trace",
        "hpa_plausibility": "plausible", "hpa_trace": "",
        "fold_evidence_status": "not_run", "fold_trace": "",
        "glycan_binding_gene": None,
    }
    contradictions = gate.find_contradictions(row, facts)
    assert len(contradictions) == 1
    assert "ALREADY_KNOWN" in contradictions[0]


def test_implausible_hpa_but_verdict_says_novel_is_flagged():
    row = {"verdict": "CONFIRMED_NOVEL"}
    facts = {
        "known_interaction_strength": "none", "known_interaction_trace": "",
        "hpa_plausibility": "implausible", "hpa_trace": "trace",
        "fold_evidence_status": "not_run", "fold_trace": "",
        "glycan_binding_gene": None,
    }
    contradictions = gate.find_contradictions(row, facts)
    assert any("IMPLAUSIBLE" in c for c in contradictions)


def test_glycan_binding_gene_not_flagged_when_verdict_already_artifact_ptm():
    # a fold clearing the residue-level PTM check must NOT override a
    # mechanism-level glycan-binding concern that's independent of it.
    row = {"verdict": "LIKELY_ARTIFACT_PTM"}
    facts = {
        "known_interaction_strength": "none", "known_interaction_trace": "",
        "hpa_plausibility": "plausible", "hpa_trace": "",
        "fold_evidence_status": "pass", "fold_trace": "trace",
        "glycan_binding_gene": "LGALS1",
    }
    contradictions = gate.find_contradictions(row, facts)
    assert contradictions == []


def test_glycan_binding_gene_flagged_when_verdict_ignores_it():
    row = {"verdict": "CONFIRMED_NOVEL"}
    facts = {
        "known_interaction_strength": "none", "known_interaction_trace": "",
        "hpa_plausibility": "plausible", "hpa_trace": "",
        "fold_evidence_status": "not_run", "fold_trace": "",
        "glycan_binding_gene": "LGALS1",
    }
    contradictions = gate.find_contradictions(row, facts)
    assert any("LGALS1" in c and "glycan" in c.lower() for c in contradictions)


# ---------------------------------------------------------------- _glycan_binding_gene (network mocked)

def test_glycan_binding_gene_fires_on_real_galectin_function_text(monkeypatch):
    def fake_fetch(gene):
        texts = {
            "LGALS1": {"function": "Lectin that binds beta-galactoside and a wide array of complex carbohydrates."},
            "CSF2RA": {"function": "Receptor for GM-CSF, a hematopoietic growth factor."},
        }
        return texts[gene]

    monkeypatch.setattr(gate, "fetch_uniprot_annotation", fake_fetch)
    assert gate._glycan_binding_gene("LGALS1", "CSF2RA") == "LGALS1"


def test_glycan_binding_gene_does_not_fire_on_plain_protein_receptor(monkeypatch):
    # the "Lectin" KEYWORD generalization was tested and rejected for being
    # too broad (fires on the whole C-type lectin DOMAIN receptor family) —
    # this guards that the FUNCTION-TEXT check stays precise.
    def fake_fetch(gene):
        texts = {
            "CLEC2B": {"function": "Membrane-bound protein which acts as a ligand to stimulate the activating receptor NKp80/KLRF1."},
            "KLRD1": {"function": "Immune receptor involved in self-nonself discrimination, recognizes HLA-E."},
        }
        return texts[gene]

    monkeypatch.setattr(gate, "fetch_uniprot_annotation", fake_fetch)
    assert gate._glycan_binding_gene("CLEC2B", "KLRD1") is None
