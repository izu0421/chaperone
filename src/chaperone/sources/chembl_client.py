"""ChEMBL and PubChem lookups for the small-molecule side of a candidate.

For a protein-protein candidate, "is this already known" is answered by
STRING/CellPhoneDB/PubMed. None of those index a metabolite against a
receptor, so a protein-ligand candidate needs its own source: ChEMBL holds
curated bioactivities (Ki, IC50, EC50) linking a molecule to a protein
target, which is the pharmacology literature's own record of the pair.

Two questions are asked here, and the second matters as much as the first:

  known_activity   does this molecule have measured activity AT this target?
  promiscuity      how many DISTINCT targets does it have activity at?

The second exists because co-folding confidence does not discriminate for a
small molecule in a pocket: in this project's own orphan screen, 12 of 21
metabolite candidates reached interface pTM 0.5 against EVERY receptor they
were folded against, where no protein partner of 419 did. A molecule that
ChEMBL already shows hitting dozens of unrelated targets is the same kind of
candidate, and the agent should see that count before it reads a confident
fold as evidence of a specific interaction.
"""
from urllib.parse import quote

import httpx

from .http_retry import get_with_retry

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 30.0
# a bioactivity worth calling "known": these are the standard affinity and
# potency endpoints, as opposed to assay-specific readouts that cannot be
# compared across papers
AFFINITY_TYPES = {"Ki", "Kd", "IC50", "EC50", "AC50", "Potency", "pKi", "pIC50", "pEC50", "pA2", "pD2"}


def _get(client, url, params):
    return get_with_retry(client, url, {**params, "format": "json"}, TIMEOUT).json()


def _chembl_id_by_inchi_key(client, inchi_key: str):
    """The ChEMBL id of a molecule identified by structure, or None."""
    if not inchi_key:
        return None
    try:
        data = _get(client, f"{CHEMBL}/molecule",
                    {"molecule_structures__standard_inchi_key": inchi_key})
    except Exception:  # noqa: BLE001  — a miss here only costs the id
        return None
    molecules = data.get("molecules") or []
    return molecules[0].get("molecule_chembl_id") if molecules else None


def resolve_molecule(name: str) -> dict:
    """A molecule name to its ChEMBL id and SMILES, falling back to PubChem.

    The fallback matters for metabolites: ChEMBL indexes drug-like compounds
    well but not every endogenous metabolite, and a candidate that cannot be
    resolved to a structure cannot be folded at all, so an unresolved name is
    reported as such instead of silently folding nothing.
    """
    ident = (name or "").strip()
    if not ident:
        return {"error": "no molecule name given"}
    with httpx.Client() as client:
        try:
            data = _get(client, f"{CHEMBL}/molecule", {"pref_name__iexact": ident})
            molecules = data.get("molecules") or []
            if molecules:
                m = molecules[0]
                structures = m.get("molecule_structures") or {}
                # a ChEMBL entry without a structure cannot be folded, so fall
                # through to PubChem rather than returning smiles=None
                if structures.get("canonical_smiles"):
                    return {
                        "query": ident,
                        "source": "chembl",
                        "molecule_chembl_id": m.get("molecule_chembl_id"),
                        "pref_name": m.get("pref_name"),
                        "smiles": structures.get("canonical_smiles"),
                        "inchi_key": structures.get("standard_inchi_key"),
                    }
        except Exception as exc:  # noqa: BLE001
            chembl_error = str(exc)
        else:
            chembl_error = None
        try:
            # PubChem takes the output format in the URL path and rejects a
            # `format` query parameter outright, so it cannot go through _get
            resp = get_with_retry(
                client, f"{PUBCHEM}/compound/name/{quote(ident)}/property/SMILES,InChIKey/JSON",
                {}, TIMEOUT)
            props = (resp.json().get("PropertyTable") or {}).get("Properties") or []
            # PubChem renamed this property: asking for CanonicalSMILES now
            # returns a key called ConnectivitySMILES, so reading one fixed
            # key returned None with no error and folded nothing at all
            smiles = next((props[0][k] for k in ("SMILES", "CanonicalSMILES", "ConnectivitySMILES")
                           if props and props[0].get(k)), None)
            if smiles:
                inchi_key = props[0].get("InChIKey")
                return {
                    "query": ident,
                    "source": "pubchem",
                    # ChEMBL indexes by its own preferred name, so a common
                    # name like "GABA" misses on name but matches on InChIKey;
                    # without this the pair would look absent from ChEMBL when
                    # it is simply filed under another name
                    "molecule_chembl_id": _chembl_id_by_inchi_key(client, inchi_key),
                    "pref_name": ident,
                    "smiles": smiles,
                    "inchi_key": inchi_key,
                }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"neither ChEMBL nor PubChem resolved '{ident}': {exc}",
                    "chembl_error": chembl_error}
    return {"error": f"no ChEMBL or PubChem record for '{ident}'", "chembl_error": chembl_error}


def _target_ids(client, accession: str) -> list:
    """ChEMBL target ids for a UniProt accession, single proteins only.

    Protein-family and selectivity-group targets are excluded: an activity
    recorded against "Serotonin (5-HT) receptor" as a family does not say
    which receptor was measured, so counting it as a hit for one member
    would manufacture a known interaction that nobody reported.
    """
    data = _get(client, f"{CHEMBL}/target", {"target_components__accession": accession})
    return [t["target_chembl_id"] for t in data.get("targets", [])
            if t.get("target_type") == "SINGLE PROTEIN"]


def known_activity(accession: str, molecule_chembl_id: str) -> dict:
    """Curated bioactivities of one molecule at one protein target."""
    if not accession or not molecule_chembl_id:
        return {"error": "both a UniProt accession and a ChEMBL molecule id are required"}
    with httpx.Client() as client:
        targets = _target_ids(client, accession)
        if not targets:
            return {"accession": accession, "molecule_chembl_id": molecule_chembl_id,
                    "target_chembl_ids": [], "n_activities": 0,
                    "note": "no ChEMBL SINGLE PROTEIN target for this accession, so ChEMBL "
                            "cannot speak to this pair either way"}
        rows, total = [], 0
        for target in targets:
            data = _get(client, f"{CHEMBL}/activity",
                        {"molecule_chembl_id": molecule_chembl_id, "target_chembl_id": target,
                         "limit": 20})
            total += (data.get("page_meta") or {}).get("total_count", 0)
            for a in data.get("activities", []):
                rows.append({
                    "target_chembl_id": target,
                    "standard_type": a.get("standard_type"),
                    "standard_value": a.get("standard_value"),
                    "standard_units": a.get("standard_units"),
                    "pchembl_value": a.get("pchembl_value"),
                    "assay_description": (a.get("assay_description") or "")[:160],
                })
    affinities = [r for r in rows if r["standard_type"] in AFFINITY_TYPES]
    return {
        "accession": accession,
        "molecule_chembl_id": molecule_chembl_id,
        "target_chembl_ids": targets,
        "n_activities": total,
        "n_affinity_measurements_shown": len(affinities),
        "activities": affinities[:10] or rows[:5],
    }


def promiscuity(molecule_chembl_id: str, max_targets: int = 400) -> dict:
    """How many distinct single-protein targets this molecule is active at.

    A confident fold against one receptor says little about a molecule that
    is already measured against a hundred of them.
    """
    if not molecule_chembl_id:
        return {"error": "a ChEMBL molecule id is required"}
    with httpx.Client() as client:
        # No `only=` field selector here. ChEMBL honours it inconsistently:
        # the same request returned rows complete with target_chembl_id from
        # one caller and rows missing it from another seconds later, which
        # made a molecule measured at 79 targets report 0 — and 0 reads as
        # "selective", the opposite of the truth, in exactly the direction
        # that would let a promiscuous binder through as a novel hit.
        data = _get(client, f"{CHEMBL}/activity",
                    {"molecule_chembl_id": molecule_chembl_id,
                     "standard_type__in": ",".join(sorted(AFFINITY_TYPES)),
                     "limit": max_targets})
    activities = data.get("activities", [])
    total = (data.get("page_meta") or {}).get("total_count", 0)
    by_target = {}
    for a in activities:
        if a.get("target_chembl_id"):
            by_target.setdefault(a["target_chembl_id"], a.get("target_pref_name"))
    if activities and not by_target:
        # measurements came back carrying no target at all: the count is
        # unusable, and saying so is not the same as saying zero
        return {"error": f"ChEMBL returned {len(activities)} activities for "
                         f"{molecule_chembl_id} with no target id on any of them, so the "
                         f"distinct-target count is unavailable — do not read this as "
                         f"a selective molecule",
                "molecule_chembl_id": molecule_chembl_id,
                "n_affinity_measurements": total,
                "measurements_sampled": len(activities)}
    return {
        "molecule_chembl_id": molecule_chembl_id,
        "n_affinity_measurements": total,
        "n_distinct_targets_sampled": len(by_target),
        "measurements_sampled": len(activities),
        "example_targets": sorted({n for n in by_target.values() if n})[:15],
        "note": "targets are counted over a sample of up to "
                f"{max_targets} affinity measurements, so this is a floor, not a total",
    }
