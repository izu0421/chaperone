"""The provenance fingerprint must catch a changed job and ignore harmless noise.

A job name is not proof of what was folded. These pin the three behaviours the
check relies on: a real sequence change is caught, a reordering of the chain list
is not mistaken for one, and MSA content is deliberately out of scope.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cp", ROOT / "scripts" / "check_fold_provenance.py")
CP = importlib.util.module_from_spec(_spec)
sys.modules["cp"] = CP
_spec.loader.exec_module(CP)

fp = CP.sequence_fingerprint

BASE = {"sequences": [
    {"protein": {"id": "A", "sequence": "MKVLA", "unpairedMsa": ">a\nMKVLA\n"}},
    {"protein": {"id": "B", "sequence": "GGWWP", "unpairedMsa": ">b\nGGWWP\n"}}]}


def swap(d, path, value):
    import copy
    out = copy.deepcopy(d)
    node = out
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return out


def test_identical_input_matches():
    assert fp(BASE) == fp(swap(BASE, ["sequences", 0, "protein", "id"], "A"))


def test_one_residue_change_is_caught():
    assert fp(swap(BASE, ["sequences", 1, "protein", "sequence"], "GGWWA")) != fp(BASE)


def test_truncation_is_caught():
    assert fp(swap(BASE, ["sequences", 1, "protein", "sequence"], "GGWW")) != fp(BASE)


def test_chain_reordering_is_not_a_change():
    reordered = {"sequences": list(reversed(BASE["sequences"]))}
    assert fp(reordered) == fp(BASE)


def test_case_and_whitespace_are_normalised():
    assert fp(swap(BASE, ["sequences", 0, "protein", "sequence"], " mkvla ")) == fp(BASE)


def test_msa_content_is_out_of_scope():
    """MSAs are large and legitimately vary between builds; flagging them flags all."""
    assert fp(swap(BASE, ["sequences", 0, "protein", "unpairedMsa"], ">z\nQQQQQ\n")) == fp(BASE)


def test_ligand_identity_is_covered():
    lig = {"sequences": [BASE["sequences"][0],
                         {"ligand": {"id": "B", "ccdCodes": ["ATP"]}}]}
    other = {"sequences": [BASE["sequences"][0],
                           {"ligand": {"id": "B", "ccdCodes": ["ADP"]}}]}
    assert fp(lig) != fp(other)
    assert fp(lig) != fp(BASE)


def test_smiles_change_is_covered():
    a = {"sequences": [{"ligand": {"id": "B", "smiles": "CCO"}}]}
    b = {"sequences": [{"ligand": {"id": "B", "smiles": "CCC"}}]}
    assert fp(a) != fp(b)


def test_chain_id_swap_is_a_change():
    """A and B carrying each other's sequence is a different complex to score."""
    swapped = {"sequences": [
        {"protein": {"id": "A", "sequence": "GGWWP"}},
        {"protein": {"id": "B", "sequence": "MKVLA"}}]}
    assert fp(swapped) != fp(BASE)
