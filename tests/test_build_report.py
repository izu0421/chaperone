"""Tests for build_report.classify_validation — the function that decides
whether a candidate is shown as "worth validating" in the report. Covers
the real regression this session found: a fold can clear the pLDDT/PTM
checks numerically while its own topology finding (fold_evidence_status)
argues the modeled complex can't be trusted at all — classify_validation
must defer to that, not a second, cruder copy of the same logic.
"""
import chaperone.build_report as br


def _row(verdict, other_subunits=None):
    return {"protein_a": "GENE_A", "protein_b": "GENE_B", "verdict": verdict, "other_subunits": other_subunits}


def _interface(chain_a="A", chain_b="B", plddt_a=90, plddt_b=90, ptm_a=None, ptm_b=None, topo_a=None, topo_b=None):
    return {
        "chain_a": chain_a, "chain_b": chain_b,
        "interface_plddt_a": {"mean": plddt_a}, "interface_plddt_b": {"mean": plddt_b},
        "ptm_sites_at_interface_a": ptm_a or [], "ptm_sites_at_interface_b": ptm_b or [],
        "interface_topology_a": topo_a or [], "interface_topology_b": topo_b or [],
    }


def _executed(*interfaces):
    return [{"structural_analysis": {"interfaces": list(interfaces)}}]


def test_confirmed_novel_needs_no_fold_evidence():
    assert br.classify_validation(_row("CONFIRMED_NOVEL"), []) is not None


def test_excluded_verdicts_are_never_recommended():
    for v in ["ALREADY_KNOWN", "IMPLAUSIBLE", "INSUFFICIENT_EVIDENCE"]:
        assert br.classify_validation(_row(v), []) is None


def test_subcomplex_confirmed_by_confident_interface():
    row = _row("LIKELY_SUBCOMPLEX", other_subunits="PARTNER")
    executed = _executed(_interface(plddt_a=80, plddt_b=75))
    result = br.classify_validation(row, executed)
    assert result is not None
    assert "confirmed" in result["note"].lower()


def test_subcomplex_not_resolved_without_other_subunits_named():
    row = _row("LIKELY_SUBCOMPLEX", other_subunits=None)
    assert br.classify_validation(row, _executed(_interface())) is None


def test_subcomplex_not_resolved_without_any_executed_fold():
    row = _row("LIKELY_SUBCOMPLEX", other_subunits="PARTNER")
    assert br.classify_validation(row, []) is None


def test_subcomplex_is_NOT_blocked_by_a_topology_violation():
    # This test asserted the opposite until 2026-11-24, on the reasoning that an
    # interface spanning both faces is impossible across an intact membrane. The
    # reasoning is sound and the rule still does not work: measured on 256 folds
    # of 16 receptors whose ligands are known, it rejects 61% of REAL binders and
    # 55% of non-binders, so passing it makes a pair LESS likely to be real than
    # the base rate. AlphaFold3 models no bilayer, and the inputs carried
    # uncleaved signal peptides that supplied every cytoplasmic contact in 10 of
    # 11 falsely flagged known binders.
    #
    # A LIKELY_SUBCOMPLEX verdict asks whether the named subunit completes the
    # interface. That is answered by re-folding with it, not by where a
    # two-chain model put the partner.
    row = _row("LIKELY_SUBCOMPLEX", other_subunits="PARTNER")
    executed = _executed(_interface(plddt_a=80, plddt_b=75, topo_a=["Cytoplasmic"], topo_b=["Extracellular"]))
    assert br.classify_validation(row, executed) is not None


def test_subcomplex_not_resolved_with_weak_interface():
    row = _row("LIKELY_SUBCOMPLEX", other_subunits="PARTNER")
    executed = _executed(_interface(plddt_a=30, plddt_b=25))
    assert br.classify_validation(row, executed) is None


def test_artifact_ptm_resolved_when_no_ptm_at_interface():
    row = _row("LIKELY_ARTIFACT_PTM")
    executed = _executed(_interface())
    result = br.classify_validation(row, executed)
    assert result is not None
    assert "resolved" in result["note"].lower()


def test_artifact_ptm_not_resolved_when_ptm_at_interface():
    row = _row("LIKELY_ARTIFACT_PTM")
    executed = _executed(_interface(ptm_a=[{"type": "Glycosylation", "start": 1, "end": 1}]))
    assert br.classify_validation(row, executed) is None


def test_artifact_ptm_is_NOT_blocked_by_a_topology_violation():
    # Same inversion, same measurement. This mattered more than the subcomplex
    # case: resolve_ptm_artifact.py returned CONFIRMED_ARTIFACT on a topology
    # violation and stopped, which sealed every LIKELY_ARTIFACT_PTM candidate by
    # refusing the one glycan-aware re-fold that could have resolved it.
    row = _row("LIKELY_ARTIFACT_PTM")
    executed = _executed(_interface(topo_a=["Cytoplasmic"], topo_b=["Extracellular"]))
    assert br.classify_validation(row, executed) is not None
