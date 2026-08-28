"""load_candidates auto-detects two real column-naming schemas."""
from chaperone.ingest import load_candidates


def test_generic_schema(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("protein_a,protein_b,iptm,ptm\nEGFR,EREG,0.82,0.71\n")
    candidates = load_candidates(str(csv_path))
    assert len(candidates) == 1
    c = candidates[0]
    assert c.protein_a == "EGFR"
    assert c.protein_b == "EREG"
    assert c.iptm == 0.82
    assert c.ptm == 0.71
    assert c.pair_id == "EGFR__EREG"


def test_ligand_receptor_schema_with_af3_scores(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text(
        "ligand,receptor,af3_iptm,af3_ptm,n_samples_colocalized\n"
        "LGALS1,CSF2RA,0.56,0.61,3\n"
    )
    candidates = load_candidates(str(csv_path))
    assert len(candidates) == 1
    c = candidates[0]
    assert c.protein_a == "LGALS1"
    assert c.protein_b == "CSF2RA"
    assert c.iptm == 0.56
    assert c.ptm == 0.61
    # real observed co-localization is a stronger signal than anything
    # HPA/STRING infer, and must survive into `extra` so the agent sees it
    assert c.extra["n_samples_colocalized"] == "3"


def test_missing_optional_fields_default_to_none(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("protein_a,protein_b\nCD69,CLEC2B\n")
    c = load_candidates(str(csv_path))[0]
    assert c.iptm is None
    assert c.ptm is None
    assert c.model_path is None


def test_malformed_numeric_field_becomes_none_not_a_crash(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("protein_a,protein_b,iptm\nA,B,not_a_number\n")
    c = load_candidates(str(csv_path))[0]
    assert c.iptm is None
