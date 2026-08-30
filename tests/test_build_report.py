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


def test_subcomplex_blocked_by_topology_violation_despite_good_plddt():
    # the exact real bug found this session (KLRK1/KLRD1): every interface
    # cleared pLDDT >= 50, but the fold also showed an interface spanning
    # both Cytoplasmic and Extracellular residues at once — topologically
    # impossible across an intact membrane. That must block resolution,
    # not just get silently outweighed by decent pLDDT numbers.
    row = _row("LIKELY_SUBCOMPLEX", other_subunits="PARTNER")
    executed = _executed(_interface(plddt_a=80, plddt_b=75, topo_a=["Cytoplasmic"], topo_b=["Extracellular"]))
    assert br.classify_validation(row, executed) is None


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


def test_artifact_ptm_blocked_by_topology_violation_even_without_ptm_at_its_own_interface():
    # a topology violation is a stronger, more fundamental problem than the
    # originally-named PTM concern - it should block resolution too, not
    # just get ignored because THIS interface happens to lack a PTM site.
    row = _row("LIKELY_ARTIFACT_PTM")
    executed = _executed(_interface(topo_a=["Cytoplasmic"], topo_b=["Extracellular"]))
    assert br.classify_validation(row, executed) is None
