"""The gate must not report "no fold was run" for a pair whose fold is
sitting on disk.

This is the regression that got through once already: the bulk pre-screen
writes its jobs to fold_runs/<a>__<b>_agentfold/, and an earlier version of
gather_fold_results only ever looked in fold_runs/followups/_summary.json
plus the conversation's own fold_complex tool calls. On this project's real
75-candidate run that meant fold evidence was found for 25 candidates
instead of 62 — the other 39 silently reported fold_evidence_status
"not_run" with 62 *_agentfold directories present. Nothing failed; the gate
just stopped checking two thirds of the folds.

These tests use a fake fold_runs tree (no GPU, no network, no real CIF), so
they assert the DISCOVERY and the status reporting, not the structural
analysis itself.
"""
import importlib
import json

import pytest


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """deterministic_gate with its bulk fold directory pointed at tmp_path."""
    monkeypatch.setenv("CHAPERONE_HOME", str(tmp_path))
    import chaperone.paths
    import chaperone.deterministic_gate as g
    importlib.reload(chaperone.paths)
    importlib.reload(g)
    assert g.BULK_FOLD_RUNS_DIR == tmp_path / "fold_runs"
    return g


def _write_agentfold(root, gene_a, gene_b, iptm=0.42):
    """Lay out a bulk pre-screen job exactly as fold_candidate.fold() does."""
    name = f"{gene_a.lower()}__{gene_b.lower()}_agentfold"
    job_dir = root / "fold_runs" / name / "output" / name
    job_dir.mkdir(parents=True)
    (job_dir / f"{name}_model.cif").write_text("data_fake\n")
    (job_dir / f"{name}_summary_confidences.json").write_text(json.dumps({"iptm": iptm, "ptm": 0.5}))
    return job_dir, name


def test_discovers_bulk_screen_fold_on_disk(gate, tmp_path, monkeypatch):
    _write_agentfold(tmp_path, "CFL1", "GRM5")
    monkeypatch.setattr(gate, "_get_accession", lambda gene: f"ACC_{gene}")
    monkeypatch.setattr(gate, "analyze", lambda cif, accs: {"interfaces": []})

    found = gate.discover_bulk_screen_fold("CFL1", "GRM5")
    assert found is not None, "a fold exists under fold_runs/*_agentfold and was not found"
    assert found["source"] == "bulk_screen"
    assert found["iptm"] == 0.42


def test_discovery_is_gene_order_independent(gate, tmp_path, monkeypatch):
    # The directory is named in the job's chain order, which need not match
    # the order the verdict row happens to list the pair in.
    _write_agentfold(tmp_path, "CFL1", "GRM5")
    monkeypatch.setattr(gate, "_get_accession", lambda gene: f"ACC_{gene}")
    monkeypatch.setattr(gate, "analyze", lambda cif, accs: {"interfaces": []})

    found = gate.discover_bulk_screen_fold("GRM5", "CFL1")
    assert found is not None
    # Chain order must follow the on-disk job, not the caller's argument order.
    assert [c["identifier"] for c in found["chains"]] == ["CFL1", "GRM5"]


def test_gate_does_not_report_not_run_when_a_fold_exists(gate, tmp_path, monkeypatch):
    """The whole point: the status the gate reports for such a pair."""
    _write_agentfold(tmp_path, "CFL1", "GRM5")
    monkeypatch.setattr(gate, "_get_accession", lambda gene: f"ACC_{gene}")
    monkeypatch.setattr(gate, "analyze", lambda cif, accs: {
        "interfaces": [{
            "chain_a": "A", "chain_b": "B",
            "ptm_sites_at_interface_a": [], "ptm_sites_at_interface_b": [],
            "interface_topology_a": ["Cytoplasmic"], "interface_topology_b": ["Extracellular"],
        }]
    })
    row = {"protein_a": "CFL1", "protein_b": "GRM5"}
    results = gate.gather_fold_results(row, tool_calls=[], fold_by_pair={})
    status = gate.fold_evidence_status(results)
    assert status["status"] == "topology_violation", (
        f"a real fold with a Cytoplasmic+Extracellular interface was reported as "
        f"{status['status']!r}"
    )


def test_no_fold_on_disk_still_reports_not_run(gate):
    row = {"protein_a": "NOSUCH", "protein_b": "GENE"}
    results = gate.gather_fold_results(row, tool_calls=[], fold_by_pair={})
    assert results == []
    assert gate.fold_evidence_status(results)["status"] == "not_run"


def test_unanalysed_fold_is_not_reported_as_pass(gate, tmp_path, monkeypatch):
    """A fold whose interface was never analysed must not read as a clean
    bill of health — "pass" is what makes find_contradictions argue a
    LIKELY_ARTIFACT_PTM verdict has been tested and cleared."""
    _write_agentfold(tmp_path, "CFL1", "GRM5")
    # UniProt unreachable / symbol unresolvable: the fold is real, the
    # analysis is not possible.
    monkeypatch.setattr(gate, "_get_accession", lambda gene: None)

    row = {"protein_a": "CFL1", "protein_b": "GRM5"}
    results = gate.gather_fold_results(row, tool_calls=[], fold_by_pair={})
    assert len(results) == 1
    status = gate.fold_evidence_status(results)
    assert status["status"] == "analysis_unavailable"

    facts = {
        "known_interaction_strength": "none", "known_interaction_trace": "",
        "hpa_plausibility": "unknown", "hpa_trace": "",
        "fold_evidence_status": status["status"], "fold_trace": status["trace"],
        "glycan_binding_gene": None,
    }
    contradictions = gate.find_contradictions({"verdict": "LIKELY_ARTIFACT_PTM"}, facts)
    assert not any("has been directly tested and not confirmed" in c for c in contradictions), (
        "an unanalysed fold was used to argue the PTM concern had been cleared"
    )


def test_cached_gate_result_is_reused(gate, tmp_path, monkeypatch):
    job_dir, name = _write_agentfold(tmp_path, "CFL1", "GRM5")
    (job_dir / f"{name}_gate_result.json").write_text(json.dumps({
        "job_name": name, "source": "bulk_screen",
        "structural_analysis": {"interfaces": []}, "iptm": 0.9,
    }))

    def _boom(*a, **k):
        raise AssertionError("re-analysed a fold that already had a cached gate result")

    monkeypatch.setattr(gate, "analyze", _boom)
    monkeypatch.setattr(gate, "_get_accession", _boom)
    found = gate.discover_bulk_screen_fold("CFL1", "GRM5")
    assert found["iptm"] == 0.9


def _cytoplasmic_only_fold(gate, tmp_path, monkeypatch):
    """A fold whose whole interface sits on the cytoplasmic face."""
    _write_agentfold(tmp_path, "SLIT2", "GPR52")
    monkeypatch.setattr(gate, "_get_accession", lambda gene: f"ACC_{gene}")
    monkeypatch.setattr(gate, "analyze", lambda cif, accs: {
        "interfaces": [{
            "chain_a": "A", "chain_b": "B",
            "ptm_sites_at_interface_a": [], "ptm_sites_at_interface_b": [],
            "interface_topology_a": ["Cytoplasmic"], "interface_topology_b": [],
        }]
    })
    row = {"protein_a": "SLIT2", "protein_b": "GPR52"}
    return gate.gather_fold_results(row, tool_calls=[], fold_by_pair={})


def test_secreted_partner_on_the_cytoplasmic_face_is_a_violation(gate, tmp_path, monkeypatch):
    """The gap the structures exposed: the both-faces rule catches an interface
    straddling the membrane, but nine of the twelve pairs the gate had just
    confirmed put a SECRETED partner wholly on the inside, which the rule said
    nothing about. A secreted protein does not reach the cytoplasm."""
    results = _cytoplasmic_only_fold(gate, tmp_path, monkeypatch)
    status = gate.fold_evidence_status(results, secreted_partner=True)
    assert status["status"] == "topology_violation", (
        f"a secreted partner docked entirely on the cytoplasmic face was reported "
        f"as {status['status']!r}"
    )
    assert "secreted" in status["trace"]


def test_same_geometry_passes_when_nothing_is_secreted(gate, tmp_path, monkeypatch):
    """The guard on the rule. A membrane protein with a cytoplasmic domain
    contacting another protein's cytoplasmic face is ordinary biology, and
    flagging it would re-create the over-firing _protein_class_flags was
    written to avoid. No secretion evidence, no flag."""
    results = _cytoplasmic_only_fold(gate, tmp_path, monkeypatch)
    assert gate.fold_evidence_status(results, secreted_partner=False)["status"] == "pass"
