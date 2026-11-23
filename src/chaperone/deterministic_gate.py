"""A code-based, reproducible gate over each candidate's REAL structured tool
outputs (STRING per-channel scores, CellPhoneDB curated hits, real HPA
expression/compartment data, real fold structural analysis) — not the LLM's
free-text verdict, which varies between runs and can't be checked
mechanically. Per-candidate LLM calls are useful for gathering and narrating
evidence, but the final tiering decision should not depend on which way an
independent LLM call happened to land on a given day.

Fixes two specific problems found in the original LLM-only verdicts:
1. STRING is sometimes wrong/misleading as "known interaction" evidence — a
   combined_score driven entirely by textmining/coexpression (no database or
   experiments channel) is a co-mention signal, not proof of a real curated
   or experimentally-observed interaction. Treating ANY STRING hit as
   disqualifying "novel" was a real bug; only a strong channel (database or
   experiments) counts as ALREADY_KNOWN now.
2. Nothing previously forced IMPLAUSIBLE when HPA genuinely showed no route
   for the two proteins to ever meet (incompatible compartments + zero
   shared tissue/cell-type expression) — that was left to the LLM's
   discretion per-candidate, inconsistently.

Usage:
    python -m chaperone.deterministic_gate data/verdicts_full.csv log \
        --fold-summary fold_runs/followups/_summary.json --out data/verdicts_gated.csv
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


from .sources.uniprot_client import fetch_uniprot_annotation  # noqa: E402
from .analyze_fold import analyze  # noqa: E402
from .fold_candidate import fetch_sequence, CHAIN_LETTERS  # noqa: E402
from .paths import PROJECT_ROOT as ROOT, FOLD_RUNS_DIR  # noqa: E402

# Where the bulk pre-screen's own fold jobs live, as '<a>__<b>_agentfold/'
# directories. Configurable with the rest of the project layout via
# CHAPERONE_HOME (see paths.py); a pip-installed chaperone pointed at a
# different project directory picks that project's folds up instead.
BULK_FOLD_RUNS_DIR = FOLD_RUNS_DIR

# Where a SCREEN's folds live, which is not where this package puts its own.
#
# discover_bulk_screen_fold looked only for "<a>__<b>_agentfold" under
# fold_runs/, a convention from an earlier bulk run. fold_runs/ is empty and the
# orphan screen writes "<A>__<B>/" somewhere else entirely, so the topology rule
# -- the one that rejects an interface spanning both membrane faces -- could not
# fire for a single candidate in that screen. It did not fail; it never ran.
#
# It would have caught a lot. Of the fourteen protein candidates the evidence
# agent supported, nine have interface residues on the cytoplasmic face and four
# span both faces at once. Fibronectin, a secreted protein, is docked with all
# thirty-one of its interface residues inside the cell.
#
# Same shape as the partner-breadth rule earlier the same day: a check reading
# from a different stage's output than the one its data was written into passes
# its own tests and does nothing. Both are now pointed at the real directory, and
# this one takes a list so the next screen adds a path instead of going unchecked.
SCREEN_FOLD_DIRS = [
    Path(d) for d in os.environ.get(
        "CHAPERONE_SCREEN_FOLD_DIRS",
        "/data/yzy21/yy/space_agent/structural_modelling/results/orphan_out:"
        "/data/yzy21/yy/space_agent/structural_modelling/results/specificity_out"
    ).split(":") if d
]

# STRING channel score at/above this is treated as real curated/experimental
# backing, not just a co-mention. STRING itself treats >=0.4 as "medium
# confidence" for a single channel; 0.3 is deliberately a bit more permissive
# here so a borderline-real hit isn't waved through as "novel", while a pure
# textmining/coexpression-driven score (with these at 0) still isn't enough.
STRING_STRONG_CHANNEL_THRESHOLD = 0.3

# A protein whose UniProt FUNCTION text describes an actual carbohydrate-
# binding mechanism (galectins, siglecs, other lectins) has a real,
# mechanism-level artifact-risk that "no PTM/glycosylation site on the
# specific interface residues AF3 modeled" does NOT clear — AF3 never
# represents glycans, so the true binding mode may not be the modeled
# bare-protein contact at all, regardless of which residues happen to touch.
#
# This used to be a hardcoded gene list (galectins only) — replaced with a
# live UniProt query because the obvious generalization, checking for the
# "Lectin" KEYWORD, was tested and found too broad: CLEC2B, SELE, and KLRD1
# (all C-type lectin DOMAIN receptors, common in this dataset) carry the
# "Lectin" keyword too, but their function text describes plain protein-
# protein recognition (e.g. KLRD1 binds HLA-E peptide complexes, CLEC2B
# binds the NKp80 receptor) with no carbohydrate-binding mechanism at all —
# flagging on the keyword would have wrongly hit the entire NK-receptor
# family already correctly handled in this dataset. Matching actual
# carbohydrate-chemistry language in the function text (verified live:
# fires on LGALS1/LGALS3/SIGLEC1's "binds beta-galactoside"/"sialic-acid
# dependent binding"/etc., does NOT fire on CLEC2B/SELE/KLRD1) is precise
# and generalizes correctly beyond one hardcoded family.
GLYCAN_BINDING_FUNCTION_PATTERN = re.compile(
    r"\b(carbohydrate|galactoside|sialic acid|glycan|mannose|fucose|"
    r"n-acetylglucosamine|n-acetylgalactosamine|glucuronic|chitin)\b",
    re.IGNORECASE,
)

SURFACE_OR_SECRETED_CLASSES = {
    "cd markers", "predicted membrane proteins", "plasma proteins",
    "g-protein coupled receptors", "transporters", "voltage-gated ion channels",
}


def _glycan_binding_gene(gene_a: str, gene_b: str):
    """Returns the gene name if either protein's real UniProt FUNCTION text
    describes a carbohydrate-binding mechanism, else None."""
    for gene in (gene_a, gene_b):
        annotation = fetch_uniprot_annotation(gene)
        function_text = annotation.get("function") or ""
        if GLYCAN_BINDING_FUNCTION_PATTERN.search(function_text):
            return gene
    return None


def _tool_outputs(tool_calls: list, tool_name: str) -> list:
    return [c["output"] for c in tool_calls if c.get("tool") == tool_name and isinstance(c.get("output"), dict)]


def _pubmed_cooccurrence_count(tool_calls: list):
    """Returns the pubmed_cooccurrence hit count if that tool was called for
    this pair, else None (not checked — different from a checked zero)."""
    for out in _tool_outputs(tool_calls, "pubmed_cooccurrence"):
        return out.get("count")
    return None


def known_interaction_strength(tool_calls: list) -> dict:
    """Returns {"strength": "confirmed"|"weak"|"none", "trace": str}.

    STRING's "database" channel is not always a specific-pair binding
    citation — it can reflect curated PATHWAY/interactome databases
    (KEGG/Reactome-style) that list genes as functionally related without a
    published paper establishing THIS pair directly binds (confirmed live:
    EFNA4/EPHB4's database=0.5 score was family/pathway-level Eph-ephrin
    annotation, not curated evidence for that specific ligand-receptor
    pairing). Per explicit user direction: a pair found ONLY in STRING, with
    no PubMed literature confirming it, is still novel and worth pursuing —
    STRING alone (even a strong channel) is not sufficient for ALREADY_KNOWN
    without at least some literature trace for this specific pair. CellPhoneDB
    is still trusted on its own (manually curated ligand-receptor/complex
    data, not a pathway-membership database)."""
    for out in _tool_outputs(tool_calls, "cellphonedb_known_interaction"):
        if out.get("found"):
            return {"strength": "confirmed", "trace": f"CellPhoneDB curated hit: {out.get('matches')}"}

    for out in _tool_outputs(tool_calls, "string_known_interaction"):
        if not out.get("found"):
            continue
        ev = out.get("evidence") or {}
        database, experiments = ev.get("database", 0), ev.get("experiments", 0)
        strong_channel = database >= STRING_STRONG_CHANNEL_THRESHOLD or experiments >= STRING_STRONG_CHANNEL_THRESHOLD
        if not strong_channel:
            return {
                "strength": "weak",
                "trace": f"STRING found combined_score={out.get('combined_score')} but database={database} experiments={experiments} "
                         f"(driven by textmining={ev.get('textmining')}/coexpression={ev.get('coexpression')} only) — "
                         f"a co-mention/coexpression signal, NOT proof of a known interaction; does not disqualify novelty",
            }
        pubmed_count = _pubmed_cooccurrence_count(tool_calls)
        if pubmed_count:
            return {
                "strength": "confirmed",
                "trace": f"STRING database={database} experiments={experiments} (>= {STRING_STRONG_CHANNEL_THRESHOLD} threshold) "
                         f"AND {pubmed_count} PubMed co-occurrence hit(s) — real curated/experimental backing WITH a literature trace, not just a database/pathway annotation",
            }
        return {
            "strength": "weak",
            "trace": f"STRING database={database} experiments={experiments} is a strong channel, but pubmed_cooccurrence "
                     f"found {'0 hits' if pubmed_count == 0 else 'was never checked'} — a STRING database/pathway "
                     f"annotation with ZERO literature trace for this specific pair is not enough to call ALREADY_KNOWN; "
                     f"still counts as novel per explicit user direction (STRING-only, no literature, stays novel)",
        }

    return {"strength": "none", "trace": "No CellPhoneDB or STRING hit."}


def _protein_class_flags(tool_calls: list, gene: str) -> dict:
    for out in _tool_outputs(tool_calls, "hpa_protein_class"):
        if out.get("gene") == gene:
            classes = {c.lower() for c in (out.get("protein_class") or [])}
            secretome = (out.get("secretome_location") or "").lower()
            # A protein tagged BOTH "predicted membrane proteins" and
            # "predicted intracellular proteins" (very common — a receptor's
            # own cytoplasmic tail routinely earns it the "intracellular" tag
            # too) is NOT a mismatch risk: its cytoplasmic domain is exactly
            # where it would meet a purely cytosolic partner (receptor
            # cytoplasmic-tail biology is one of the most common real PPI
            # types there is). An earlier version of this function treated
            # ANY membrane tag as "surface_or_secreted" and flagged this
            # extremely common, usually-plausible combination as a
            # compartment mismatch (caught live: NPR3+RAB8B, LDLR+FERMT2,
            # GNAQ+GRIK4 were all wrongly flagged this way). Only a genuinely
            # SECRETED signal (leaves the cell into the extracellular space)
            # is a strong enough claim to weigh against a purely intracellular
            # partner with zero membrane presence at all.
            secreted = bool("secreted" in classes or "predicted secreted proteins" in classes or "secreted" in secretome)
            has_any_membrane_presence = bool(classes & SURFACE_OR_SECRETED_CLASSES or "membrane" in secretome)
            # "and not secreted" matters: HPA routinely tags one protein
            # BOTH "predicted secreted proteins" and "predicted
            # intracellular proteins" (measured on this project's own data:
            # 5 of 107 genes — TGM2, SMOC2, COL21A1, VWA1, ADAMTSL5), and
            # SURFACE_OR_SECRETED_CLASSES deliberately holds no secreted
            # class, so without this a genuinely secreted protein could be
            # designated "purely intracellular, no membrane presence at
            # all" in a compartment-mismatch trace against another secreted
            # partner it can perfectly well meet extracellularly.
            intracellular_only = bool(
                any("intracellular" in c for c in classes)
                and not has_any_membrane_presence
                and not secreted
            )
            return {"secreted": secreted, "intracellular_only": intracellular_only, "classes": classes}
    return {"secreted": False, "intracellular_only": False, "classes": set()}


def _protein_tissue_keys(tool_calls: list, gene: str) -> set:
    """Protein-level (antibody/IHC-derived) tissue presence, per HPA's
    'Protein tissue specific Intensity' field — deliberately NOT the RNA
    nTPM/nCPM fields. RNA presence doesn't guarantee real protein presence,
    and using it produced a false positive on real project data: NPR3's and
    RAB8B's RNA tissue-specificity fields showed zero overlap (kidney vs
    bone marrow — each gene's single most RNA-enriched tissue), but the
    protein-level field shows both genuinely present in lymphoid tissue."""
    for out in _tool_outputs(tool_calls, "hpa_expression"):
        if out.get("gene") == gene:
            return set((out.get("protein_tissue_specific_intensity") or {}).keys())
    return set()


def hpa_plausibility(tool_calls: list, gene_a: str, gene_b: str) -> dict:
    """Returns {"plausibility": "plausible"|"implausible"|"unknown", "trace": str}.
    Deliberately conservative (learned from an earlier over-firing heuristic
    in this project): only fires "implausible" when there's BOTH a real
    compartment mismatch AND zero coexpression signal — proteins do
    moonlight/shuttle, so this needs two independent red flags, not one.

    Also requires the pair to actually share an HPA protein-level tissue
    even when there's no compartment mismatch — two compatible but
    never-coexpressed proteins can't physically interact either. Exempt if
    either protein is secreted: a secreted ligand circulates and doesn't
    need to be expressed in the same tissue as its receptor (paracrine/
    endocrine signalling) — a blanket same-tissue filter would wrongly kill
    real ligand-receptor pairs like that."""
    flags_a = _protein_class_flags(tool_calls, gene_a)
    flags_b = _protein_class_flags(tool_calls, gene_b)
    if not flags_a["classes"] or not flags_b["classes"]:
        return {"plausibility": "unknown", "trace": "HPA protein_class missing for one or both genes — can't judge compartment compatibility."}

    mismatch = (
        (flags_a["intracellular_only"] and flags_b["secreted"])
        or (flags_b["intracellular_only"] and flags_a["secreted"])
    )
    tissues_a, tissues_b = _protein_tissue_keys(tool_calls, gene_a), _protein_tissue_keys(tool_calls, gene_b)
    shared_tissue = tissues_a & tissues_b

    if not mismatch:
        if flags_a["secreted"] or flags_b["secreted"]:
            return {
                "plausibility": "plausible",
                "trace": "No hard compartment mismatch, and at least one protein is secreted — paracrine/endocrine "
                         "signalling doesn't require same-tissue expression.",
            }
        if shared_tissue:
            return {
                "plausibility": "plausible",
                "trace": f"No hard compartment mismatch, and real HPA protein-level tissue overlap exists "
                         f"(shared tissue={shared_tissue}).",
            }
        if not tissues_a or not tissues_b:
            return {
                "plausibility": "unknown",
                "trace": "No hard compartment mismatch, but HPA protein-level tissue data is missing for one or "
                         "both genes — can't judge tissue coexpression.",
            }
        return {
            "plausibility": "implausible",
            "trace": "No hard compartment mismatch, but neither protein is secreted (a direct contact would need "
                     "them physically in the same place) and zero shared HPA protein-level tissue — no plausible "
                     "route for these two to meet.",
        }

    if shared_tissue:
        return {
            "plausibility": "plausible",
            "trace": f"Compartment mismatch flagged ({flags_a['classes']} vs {flags_b['classes']}) but real HPA "
                     f"protein-level tissue overlap exists (shared tissue={shared_tissue}) — proteins can "
                     f"moonlight/shuttle, treating as plausible.",
        }
    intracellular_gene, secreted_gene = (gene_a, gene_b) if flags_a["intracellular_only"] else (gene_b, gene_a)
    return {
        "plausibility": "implausible",
        "trace": f"{intracellular_gene} has no membrane presence at all per HPA (purely intracellular) and "
                 f"{secreted_gene} is genuinely secreted, AND zero shared HPA protein-level tissue — no plausible "
                 f"route for these two to physically meet.",
    }


def fold_evidence_status(fold_results: list) -> dict:
    """Returns {"status": "pass"|"ptm_only_concern"|"topology_violation"|"not_run", "trace": str}.
    Distinguishes two different-severity concerns rather than lumping them:
    a PTM/glycosylation site at the interface is a "might be gated/blocked in
    reality" concern (LIKELY_ARTIFACT_PTM territory — could still be a real
    interaction once resolved), but an interface spanning BOTH Cytoplasmic
    AND Extracellular residues is a stronger, more fundamental problem — the
    modeled complex can't exist as modeled across an intact membrane, which
    is IMPLAUSIBLE territory, not just an artifact concern. Conflating these
    (an earlier version of this function did) produced false contradictions
    against real_with_fold.py's prior, already-correct IMPLAUSIBLE calls."""
    if not fold_results:
        return {"status": "not_run", "trace": "No real fold was executed for this candidate."}

    # Only glycosylation/lipidation are the kind of modification AF3's
    # bare-sequence fold genuinely cannot represent (a branched glycan tree
    # or lipid anchor). A disulfide bond or a simple modified residue
    # (e.g. phosphorylation) is usually structurally well-represented by
    # AF3, and treating any of them as an "artifact risk" over-fired: caught
    # live when KLRK1/CD69 was flagged for a plain disulfide bond at the
    # interface, not an unmodeled glycan.
    ARTIFACT_RISK_PTM_TYPES = {"Glycosylation", "Lipidation"}
    ptm_notes, topology_notes = [], []
    for r in fold_results:
        for iface in (r.get("structural_analysis") or {}).get("interfaces", []):
            ptm_a = [p for p in (iface.get("ptm_sites_at_interface_a") or []) if p.get("type") in ARTIFACT_RISK_PTM_TYPES]
            ptm_b = [p for p in (iface.get("ptm_sites_at_interface_b") or []) if p.get("type") in ARTIFACT_RISK_PTM_TYPES]
            if ptm_a or ptm_b:
                ptm_notes.append(f"Glycosylation/lipidation site sits directly on the modeled {iface['chain_a']}-{iface['chain_b']} interface")
            topo = set(iface.get("interface_topology_a") or []) | set(iface.get("interface_topology_b") or [])
            if "Cytoplasmic" in topo and "Extracellular" in topo:
                topology_notes.append(f"{iface['chain_a']}-{iface['chain_b']} interface spans both Cytoplasmic and Extracellular residues — topologically impossible across an intact membrane")

    if topology_notes:
        return {"status": "topology_violation", "trace": "Real fold executed and flagged: " + "; ".join(topology_notes + ptm_notes)}
    if ptm_notes:
        return {"status": "ptm_only_concern", "trace": "Real fold executed and flagged: " + "; ".join(ptm_notes)}

    # A fold with no structural_analysis at all was never actually checked
    # for PTM sites or topology — reporting "pass" for it would be a false
    # clean bill of health, and "pass" is specifically what makes
    # find_contradictions argue a LIKELY_ARTIFACT_PTM verdict has been
    # tested and cleared. Say so instead. This became reachable once
    # discover_bulk_screen_fold started returning bulk pre-screen folds,
    # whose analysis needs a UniProt accession lookup that can fail.
    if not any((r.get("structural_analysis") or {}).get("interfaces") is not None for r in fold_results):
        errors = [r["structural_analysis_error"] for r in fold_results if r.get("structural_analysis_error")]
        return {
            "status": "analysis_unavailable",
            "trace": "A real fold exists for this candidate but its interface was never analysed, so no "
                     "PTM/topology conclusion can be drawn from it"
                     + (f" ({'; '.join(errors)})" if errors else "")
                     + " — this is NOT a clean result.",
        }
    return {"status": "pass", "trace": "Real fold executed; no PTM/glycosylation site at the interface and no membrane-topology violation."}



# A molecule this many distinct protein targets deep in ChEMBL will fold
# confidently into one more pocket whatever that pocket is. The number is a
# flag for reconsideration, not an auto-relabel: the gate's own rule is that
# code flags what is checkable and an LLM call decides the replacement.
PROMISCUOUS_TARGET_COUNT = 20


def ligand_evidence_status(tool_calls: list) -> dict:
    """What the ChEMBL tools actually returned for a small-molecule partner.

    Returns `checked=False` when they were never called, which is itself a
    finding: for a small-molecule candidate the fold's confidence cannot
    establish specificity, so a verdict reached without them rests on the one
    signal that does not discriminate.
    """
    activity = _tool_outputs(tool_calls, "chembl_known_activity")
    promiscuity = _tool_outputs(tool_calls, "chembl_ligand_promiscuity")
    resolved = _tool_outputs(tool_calls, "resolve_small_molecule")
    n_activities = max(
        [o.get("n_activities", 0) for o in activity if isinstance(o, dict)] or [0])
    # A promiscuity call that errored, or that returned no usable target
    # count, is NOT a count of zero: zero reads as "selective", which is the
    # one direction that would wave a promiscuous molecule through.
    target_counts = [o["n_distinct_targets_sampled"] for o in promiscuity
                     if isinstance(o, dict) and not o.get("error")
                     and o.get("n_distinct_targets_sampled") is not None]
    n_targets = max(target_counts) if target_counts else None
    trace = []
    if activity:
        trace.append(f"chembl_known_activity: {n_activities} measured activities at this receptor")
    if n_targets is not None:
        trace.append(f"chembl_ligand_promiscuity: active at {n_targets} distinct targets")
    elif promiscuity:
        trace.append("chembl_ligand_promiscuity returned no usable target count")
    if not resolved and not activity and not promiscuity:
        trace.append("no ChEMBL/PubChem tool was called for the small-molecule partner")
    return {
        "checked": bool(activity) and n_targets is not None,
        "resolved": bool(resolved),
        "n_activities_at_target": n_activities,
        "n_distinct_targets": n_targets,
        "promiscuous": n_targets is not None and n_targets >= PROMISCUOUS_TARGET_COUNT,
        "trace": "; ".join(trace),
    }


# A PROTEIN partner has the same failure mode as a promiscuous small molecule,
# and nothing here was checking for it. `ligand_evidence_status` catches a
# molecule that ChEMBL shows binding 20+ targets, but a protein that folds
# confidently against many receptors in the SAME screen got no equivalent
# treatment, because every verdict is reached per pair and never sees the
# partner's other folds. Caught live: PPIA (cyclophilin A) accounted for 4 of
# the 22 CONFIRMED_NOVEL calls on the orphan shortlist while being confident
# against 4 of the 11 receptors it was folded with — the fold is evidence about
# PPIA, not about any one of those receptors.
#
# The screen knows this and the triage agent does not, so the numbers are
# supplied on the candidate row rather than fetched: `partner_receptors_tested`
# and `partner_receptors_confident`. Absent, the check reports checked=False
# instead of a reassuring zero.
PROMISCUOUS_PARTNER_FRACTION = 0.30
PROMISCUOUS_PARTNER_MIN_TESTED = 5


def partner_breadth_status(row: dict) -> dict:
    """How many other receptors in the same screen this protein partner binds."""
    def _int(key):
        v = row.get(key)
        if v in (None, "", "nan"):
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    # A breadth count is only meaningful at a threshold that discriminates on the
    # arm it came from. Caught live, and it produced a wrong verdict: the counts
    # supplied for metabolite candidates had been computed at interface pTM >= 0.5,
    # which passes 97% of all small-molecule folds, so noradrenaline arrived as
    # "confident against 11 of 11 receptors". The agent used that faithfully and
    # concluded the fold was "not specific to GPR85". At the threshold calibrated
    # on receptors with known ligands it is 2 of 11. The caller must supply counts
    # taken at an arm-appropriate bar, and say which, or the field is refused.
    tested, confident = _int("partner_receptors_tested"), _int("partner_receptors_confident")
    bar = row.get("partner_breadth_threshold")
    if tested is not None and confident is not None and tested > 0 and not bar:
        return {"checked": False, "promiscuous": False,
                "trace": "breadth counts were supplied without naming the threshold they "
                         "were counted at (partner_breadth_threshold), so they cannot be "
                         "interpreted: a count taken at a bar that passes most folds "
                         "calls every partner promiscuous"}
    if tested is None or confident is None or tested <= 0:
        return {"checked": False, "promiscuous": False,
                "trace": "the partner's cross-receptor hit rate was not supplied on the "
                         "candidate row, so breadth could not be checked"}
    frac = confident / tested
    return {
        "checked": tested >= PROMISCUOUS_PARTNER_MIN_TESTED,
        "receptors_tested": tested,
        "receptors_confident": confident,
        "fraction": round(frac, 3),
        "promiscuous": tested >= PROMISCUOUS_PARTNER_MIN_TESTED
                       and frac >= PROMISCUOUS_PARTNER_FRACTION,
        "threshold": bar,
        "trace": f"this partner was folded against {tested} receptors in the same screen "
                 f"and reached {bar} against {confident} of them ({frac:.0%})",
    }


# Where the molecule sits, not how sure the model is about it.
#
# A permissive pocket is not disqualifying -- AlphaFold3 puts a small molecule
# somewhere in a receptor whatever receptor it is given, and a receptor really can
# have several ligands. What disqualifies a pose is being in the wrong place. On
# the 256-fold benchmark (16 receptors folded against all 16 panel ligands, so 16
# cells have a known-correct answer) the pose separates the two:
#
#   enclosure   every one of the 16 cognate ligands is buried at least 0.928 of its
#               own surface. A cut at 0.92 keeps 16/16 and removes 62% of the 189
#               clean negatives -- the single most useful pose criterion, and it
#               ranks as well (AUROC 0.801) as it filters.
#   wrong side  no cognate ligand contacts a cytoplasmic domain; 38% of negatives
#               do. As a veto that is free, but it is ALMOST ENTIRELY SUBSUMED by
#               enclosure: adding it to the 0.92 cut removes one further negative
#               out of 189, because a molecule on the cytoplasmic face is usually
#               poorly enclosed anyway. It is kept as a reason code, not as a
#               second independent filter, and should not be reported as one.
#
# Both are measured by analyze_fold.ligand_geometry, from UniProt topology rather
# than guessed from coordinates, so a fold that was never run returns checked=False
# instead of a pose that looks acceptable.
COGNATE_MIN_BURIAL = 0.92


def ligand_pose_status(fold_results: list) -> dict:
    """Is the small molecule enclosed, and on the right side of the membrane?"""
    poses = []
    for fr in fold_results or []:
        for name, ch in (fr.get("chains") or {}).items():
            g = ch.get("ligand_geometry") or {}
            if g.get("buried_fraction") is not None:
                poses.append((name, g))
    if not poses:
        return {"checked": False, "enclosed": None, "wrong_side": None,
                "trace": "no executed fold with a measured ligand pose, so where the "
                         "molecule sits is unknown -- not acceptable"}
    name, g = max(poses, key=lambda x: x[1].get("buried_fraction") or 0)
    burial = float(g["buried_fraction"])
    topo = []
    for pk in (g.get("pocket") or {}).values():
        topo += [str(t) for t in (pk.get("topology") or [])]
    wrong_side = any("Cytoplasm" in t for t in topo)
    return {
        "checked": True,
        "buried_fraction": round(burial, 3),
        "enclosed": burial >= COGNATE_MIN_BURIAL,
        "wrong_side": wrong_side,
        "clash": g.get("clash"),
        "trace": (f"chain {name} is buried {burial:.3f} of its own surface against a "
                  f"minimum of {min(0.928, COGNATE_MIN_BURIAL):.3f} across all 16 known "
                  f"cognate ligands"
                  + ("; and it contacts a cytoplasmic domain, which no cognate ligand "
                     "does" if wrong_side else "")),
    }


def compute_gate_facts(row: dict, tool_calls: list, fold_results: list) -> dict:
    gene_a, gene_b = row["protein_a"], row["protein_b"]
    is_ligand = (row.get("partner_kind") or "protein") == "ligand"
    known = known_interaction_strength(tool_calls)
    # the protein-protein plausibility and glycan checks compare two genes;
    # for a small-molecule partner gene_b is a molecule name, so running them
    # would query the protein tools for a gene that does not exist
    hpa = ({"plausibility": "not_applicable", "trace": "small-molecule partner"}
           if is_ligand else hpa_plausibility(tool_calls, gene_a, gene_b))
    fold = fold_evidence_status(fold_results)
    glycan_binding_gene = None if is_ligand else _glycan_binding_gene(gene_a, gene_b)
    ligand = ligand_evidence_status(tool_calls) if is_ligand else None
    pose = ligand_pose_status(fold_results) if is_ligand else None
    breadth = None if is_ligand else partner_breadth_status(row)
    facts = {"known_interaction_strength": known["strength"], "known_interaction_trace": known["trace"],
             "hpa_plausibility": hpa["plausibility"], "hpa_trace": hpa["trace"],
             "fold_evidence_status": fold["status"], "fold_trace": fold["trace"],
             "glycan_binding_gene": glycan_binding_gene,
             "partner_kind": "ligand" if is_ligand else "protein"}
    if ligand:
        facts["ligand"] = ligand
    if breadth:
        facts["partner_breadth"] = breadth
    if pose:
        facts["ligand_pose"] = pose
    return facts


def find_contradictions(row: dict, facts: dict) -> list[str]:
    """Only flag SPECIFIC, checkable contradictions between the original LLM
    verdict and hard evidence — do not attempt to auto-relabel every
    candidate a code-only rule "disagrees" with. Returns ALL applicable
    contradictions, not just the first match: an earlier version returned a
    single string and stopped at the first hit, which meant a candidate with
    BOTH a galectin-mechanism concern AND a fresh structural fold finding
    only ever had the galectin concern surfaced to the reconsideration call
    — the fold evidence (often the most specific, most decisive evidence)
    was silently never shown to the model. Caught live: LGALS1/CSF2RA's
    freshly-executed subunit-complete fold (with CSF2RB) found a real
    topology violation, but the reconsideration call never saw it because
    the galectin check fired first and the function returned immediately.

    A first version of this whole gate tried to fully replace the verdict
    changed 62/75 candidates (itself a red flag per this project's own rule:
    verify a heuristic's firing rate before trusting it) — most of that
    churn was wrongly treating "no fold was run" as "the PTM/artifact
    concern is cleared," silently discarding real PubMed-literature-based
    PTM reasoning the LLM had no fold to begin with. A second failure mode
    found live: IGHG1/FCGR2B (a textbook antibody-Fc-receptor pair) is
    genuinely ALREADY_KNOWN, but neither STRING nor CellPhoneDB happens to
    cover it — treating "our databases don't have it" as proof of "not
    known" would have wrongly reversed a correct call. So: flag only what's
    actually checkable, and resolve flagged contradictions with a real
    reconsideration LLM call (same pattern as reconsider_with_fold.py)
    rather than guessing the replacement label in code."""
    verdict = row.get("verdict")
    contradictions = []

    # ---- small-molecule partner -------------------------------------------
    ligand = facts.get("ligand")
    if ligand:
        if ligand["n_activities_at_target"] > 0 and verdict != "ALREADY_KNOWN":
            contradictions.append(
                f"ChEMBL records {ligand['n_activities_at_target']} measured bioactivities of this "
                f"molecule AT this receptor — curated pharmacology for this exact pair, the "
                f"small-molecule equivalent of a CellPhoneDB hit — but verdict is {verdict}, not "
                f"ALREADY_KNOWN. {ligand['trace']}")
        if ligand["promiscuous"] and verdict == "CONFIRMED_NOVEL":
            contradictions.append(
                f"Verdict is CONFIRMED_NOVEL, but ChEMBL already shows this molecule active at "
                f"{ligand['n_distinct_targets']} distinct protein targets. A small molecule folds "
                f"confidently into a receptor pocket almost regardless of which receptor it is given "
                f"(12 of 21 metabolites in this project's own screen cleared interface pTM 0.5 against "
                f"EVERY receptor tested, where 0 of 419 proteins did), so a confident pose for a "
                f"promiscuous molecule is the expected result and not evidence of a specific "
                f"interaction — LIKELY_PROMISCUOUS_POCKET fits this evidence. {ligand['trace']}")
        if not ligand["checked"] and verdict in ("CONFIRMED_NOVEL", "ALREADY_KNOWN", "IMPLAUSIBLE"):
            contradictions.append(
                f"Verdict is {verdict} for a small-molecule partner, but neither "
                f"chembl_known_activity nor chembl_ligand_promiscuity was called, so neither whether "
                f"this pair is already measured nor whether the molecule binds everything was "
                f"checked. The fold's confidence cannot settle either question. {ligand['trace']}")

    if facts.get("glycan_binding_gene") and verdict not in ("LIKELY_ARTIFACT_PTM", "IMPLAUSIBLE", "ALREADY_KNOWN"):
        contradictions.append(
            f"{facts['glycan_binding_gene']}'s real UniProt FUNCTION text describes a carbohydrate-binding "
            f"mechanism (e.g. \"binds beta-galactoside\", \"sialic-acid dependent binding\") — its "
            f"physiological binding is fundamentally glycan-mediated, and AF3 cannot model glycans at all, "
            f"so 'no PTM/glycosylation site landed on the specific interface residues AF3 modeled' does "
            f"NOT clear this concern (the true binding mode may not even be the modeled protein-protein "
            f"contact at all). Verdict is {verdict}, not LIKELY_ARTIFACT_PTM."
        )

    if facts["known_interaction_strength"] == "confirmed" and verdict != "ALREADY_KNOWN":
        contradictions.append(f"CellPhoneDB/STRING shows a real curated or experimental known interaction, but verdict is {verdict}, not ALREADY_KNOWN. {facts['known_interaction_trace']}")

    if verdict == "ALREADY_KNOWN" and facts["known_interaction_strength"] == "weak":
        contradictions.append(f"Verdict is ALREADY_KNOWN but the only STRING hit found is weak (textmining/coexpression-driven, no database/experiments backing) — a co-mention signal, not proof of a known interaction. {facts['known_interaction_trace']}")

    if facts["hpa_plausibility"] == "implausible" and verdict != "IMPLAUSIBLE":  # noqa: E501
        contradictions.append(f"HPA shows no plausible route for these two proteins to physically meet, but verdict is {verdict}, not IMPLAUSIBLE. {facts['hpa_trace']}")

    # Deliberately does NOT try to force IMPLAUSIBLE vs LIKELY_ARTIFACT_PTM
    # here — this project already established (see build_report.py's
    # topology-flag rendering) that AF3 not modeling a membrane means a
    # topology violation is informative, not automatic proof either way;
    # the real choice between these two often hinges on factors a rule can't
    # see (pLDDT, whether a clean alternative subunit was found instead,
    # etc.) — that's exactly what reconsider_with_fold.py's LLM call is for.
    # Only flag when the verdict shows NO acknowledgment of ANY concern
    # despite a real, checkable structural problem.
    if facts["fold_evidence_status"] in ("topology_violation", "ptm_only_concern") and verdict not in ("LIKELY_ARTIFACT_PTM", "IMPLAUSIBLE"):
        contradictions.append(f"A real AF3 fold found a real structural concern ({facts['fold_evidence_status']}), but verdict is {verdict}, which doesn't acknowledge it. {facts['fold_trace']}")

    # Don't suggest downgrading from LIKELY_ARTIFACT_PTM just because a fold's
    # SPECIFIC modeled residues happened not to overlap an annotated PTM site
    # — if the protein's own UniProt function text describes a fundamentally
    # glycan-mediated mechanism, that concern is independent of, and stronger
    # than, any one fold's residue-level finding (caught live: LGALS1/CLEC2D
    # would otherwise have been pushed to downgrade off LIKELY_ARTIFACT_PTM
    # despite LGALS1 being a genuine carbohydrate-binding lectin).
    # ---- a small molecule in the wrong place -------------------------------
    pose = facts.get("ligand_pose")
    if pose and pose.get("checked") and verdict in ("CONFIRMED_NOVEL", "ALREADY_KNOWN"):
        if not pose["enclosed"]:
            contradictions.append(
                f"The verdict is {verdict}, but the molecule is not enclosed the way a "
                f"ligand in an orthosteric pocket is: {pose['trace']}. Every one of the "
                f"16 known cognate ligands in the benchmark is buried at least 0.928; a "
                f"pose below that is resting on a surface, not sitting in a site, and "
                f"the fold's confidence does not see the difference.")
        elif pose.get("wrong_side"):
            contradictions.append(
                f"The verdict is {verdict} and the molecule is enclosed, but {pose['trace']}. "
                f"A ligand that enters from outside should not line the intracellular "
                f"face. Decide whether this is a real second site or a misplaced pose.")

    # ---- a protein partner that binds half the screen ----------------------
    # Flagged, not relabelled: a partner CAN have many real receptors (a
    # chemokine, a shared co-receptor), so the replacement label is a judgement
    # the reconsideration call makes, with the number in front of it.
    breadth = facts.get("partner_breadth")
    if (breadth and breadth.get("promiscuous")
            and verdict in ("CONFIRMED_NOVEL", "ALREADY_KNOWN")):
        contradictions.append(
            f"The verdict is {verdict}, which treats this fold as evidence about THIS "
            f"receptor, but {breadth['trace']}. A partner confident against that share of "
            f"a receptor panel is telling us about the partner, in the same way a "
            f"small molecule active at many targets is: the fold does not establish "
            f"specificity for this pair. Decide whether the breadth is biologically "
            f"real (a shared co-receptor or a chemokine with many receptors) or "
            f"whether it makes this one pair uninformative.")

    if facts["fold_evidence_status"] == "pass" and verdict == "LIKELY_ARTIFACT_PTM" and not facts.get("glycan_binding_gene"):
        contradictions.append(f"A real AF3 fold was executed specifically to check the PTM/topology concern behind this LIKELY_ARTIFACT_PTM verdict, and found no problem — the concern this verdict is based on has been directly tested and not confirmed. {facts['fold_trace']}")

    return contradictions


_accession_cache: dict = {}


def _get_accession(gene: str):
    """Cached gene->UniProt-accession lookup (network call the first time a
    gene is seen in this process)."""
    if gene not in _accession_cache:
        try:
            _accession_cache[gene] = fetch_sequence(gene)["accession"]
        except Exception:  # noqa: BLE001
            _accession_cache[gene] = None
    return _accession_cache[gene]


def ensure_structural_analysis(fold_result: dict, cache_path: Path = None) -> dict:
    """Two real gaps this closes, found auditing CLIC3/CD300A and
    CXCR1/GNAQ (a real fold existed on disk but fold_evidence_status never
    saw it):
    1. Bulk pre-screen fold_runs/<pair>_agentfold/ results (~85% of
       candidates) were never even loaded by gather_fold_results.
    2. Even fold_runs/followups/ results — the ones that ARE loaded —
       often never got structural_analysis attached (run_follow_up_folds
       computes it in-memory per job, but roughly half the *_result.json
       files on disk predate that code and were never backfilled, and
       _summary.json never carried the field at all).
    This computes (or loads a cached) structural_analysis for any fold
    result that has a model_cif + chain accessions but is missing it, and
    persists the result to cache_path (or back into a *_result.json this
    dict was loaded from) so it's not recomputed on every gate run."""
    if fold_result.get("structural_analysis", {}).get("interfaces") is not None:
        return fold_result
    model_cif = fold_result.get("model_cif")
    chains = fold_result.get("chains")
    if not model_cif or not Path(model_cif).exists() or not chains:
        return fold_result
    chain_accessions = {
        CHAIN_LETTERS[i]: c["accession"] for i, c in enumerate(chains) if c.get("accession")
    }
    if not chain_accessions:
        return fold_result
    try:
        fold_result["structural_analysis"] = analyze(model_cif, chain_accessions)
    except Exception as exc:  # noqa: BLE001
        fold_result["structural_analysis_error"] = str(exc)
        return fold_result
    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps(fold_result, indent=2))
        except Exception:  # noqa: BLE001
            pass
    return fold_result


def discover_bulk_screen_fold(gene_a: str, gene_b: str):
    """Find the original bulk pre-screen AF3/AF2M fold for this pair (job
    dirs named '<a>__<b>_agentfold', either gene order, directly under
    BULK_FOLD_RUNS_DIR — NOT its followups/ subdirectory, which is a
    separate, much smaller set gather_fold_results already checked).
    Returns a fold_result-shaped dict (with structural_analysis attached and
    cached) or None if no such directory/model exists."""
    name = job_dir = None
    # the screen's own layout first: <ROOT>/<A>__<B>/<A>__<B>_model.cif, gene case
    # as the manifest wrote it
    for root in SCREEN_FOLD_DIRS:
        if name:
            break
        for a, b in ((gene_a, gene_b), (gene_b, gene_a)):
            cand = f"{a}__{b}"
            d = root / cand
            if (d / f"{cand}_model.cif").exists():
                name, job_dir, first_gene, second_gene = cand, d, a, b
                break
    if name is not None:
        cache_path = job_dir / f"{name}_gate_result.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                pass

    for a, b in ((gene_a, gene_b), (gene_b, gene_a)):
        if name:
            break
        candidate_name = f"{a.lower()}__{b.lower()}_agentfold"
        candidate_dir = BULK_FOLD_RUNS_DIR / candidate_name / "output" / candidate_name
        if (candidate_dir / f"{candidate_name}_model.cif").exists():
            # Chain order MUST match the job's actual chain order
            # (A=first_gene, B=second_gene), which is whichever of
            # (gene_a,gene_b)/(gene_b,gene_a) matched the on-disk directory
            # name — not necessarily the caller's gene_a/gene_b order.
            name, job_dir, first_gene, second_gene = candidate_name, candidate_dir, a, b
            break
    if name is None:
        return None

    cache_path = job_dir / f"{name}_gate_result.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            pass

    summary_path = job_dir / f"{name}_summary_confidences.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    acc_first, acc_second = _get_accession(first_gene), _get_accession(second_gene)
    if not acc_first or not acc_second:
        # A fold genuinely exists on disk; we just can't resolve the
        # accessions needed to analyse it (UniProt unreachable, renamed
        # symbol). Report it WITHOUT structural_analysis so
        # fold_evidence_status says "analysis_unavailable" rather than
        # silently falling back to "not_run" — the distinction matters,
        # because this fold could still carry a topology violation.
        return {
            "job_name": name,
            "model_cif": str(job_dir / f"{name}_model.cif"),
            "chains": [{"identifier": first_gene, "accession": acc_first},
                       {"identifier": second_gene, "accession": acc_second}],
            "iptm": summary.get("iptm"),
            "ptm": summary.get("ptm"),
            "source": "bulk_screen",
            "source_pair": f"{gene_a}/{gene_b}",
            "structural_analysis_error": "could not resolve UniProt accessions for both chains",
        }
    fold_result = {
        "job_name": name,
        "model_cif": str(job_dir / f"{name}_model.cif"),
        "chains": [{"identifier": first_gene, "accession": acc_first},
                   {"identifier": second_gene, "accession": acc_second}],
        "iptm": summary.get("iptm"),
        "ptm": summary.get("ptm"),
        "source": "bulk_screen",
        "source_pair": f"{gene_a}/{gene_b}",
    }
    return ensure_structural_analysis(fold_result, cache_path=cache_path)


def load_tool_calls(row: dict, log_dir: Path) -> list:
    log_path = log_dir / f"{row['protein_a']}__{row['protein_b']}.json"
    if not log_path.exists():
        return []
    try:
        return json.loads(log_path.read_text()).get("_tool_calls", [])
    except json.JSONDecodeError:
        return []


def load_fold_results(fold_summary_path: Path) -> dict:
    if not fold_summary_path.exists():
        return {}
    by_pair = {}
    for r in json.loads(fold_summary_path.read_text()):
        by_pair.setdefault(r.get("source_pair"), []).append(r)
    return by_pair


def gather_fold_results(row: dict, tool_calls: list, fold_by_pair: dict) -> list:
    pair_key = f"{row['protein_a']}/{row['protein_b']}"
    # followups/_summary.json entries never carry structural_analysis (the
    # per-job *_result.json files are the source of truth for that, and
    # roughly half predate the field existing) — backfill+cache it here so
    # it only has to be computed once per job, ever.
    results = [
        ensure_structural_analysis(
            r, cache_path=BULK_FOLD_RUNS_DIR / "followups" / f"{r.get('job_name')}_result.json"
        )
        for r in fold_by_pair.get(pair_key, [])
    ]
    for call in tool_calls:
        if call.get("tool") == "fold_complex" and isinstance(call.get("output"), dict) and "iptm" in call["output"]:
            results.append(call["output"])
    bulk = discover_bulk_screen_fold(row["protein_a"], row["protein_b"])
    if bulk:
        results.append(bulk)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("log_dir")
    ap.add_argument("--fold-summary", default=str(ROOT / "fold_runs" / "followups" / "_summary.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "gate_facts.json"))
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    log_dir = Path(args.log_dir)
    fold_by_pair = load_fold_results(Path(args.fold_summary))

    results = []
    n_flagged = 0
    for row in rows:
        tool_calls = load_tool_calls(row, log_dir)
        fold_results = gather_fold_results(row, tool_calls, fold_by_pair)
        facts = compute_gate_facts(row, tool_calls, fold_results)
        contradictions = find_contradictions(row, facts)
        if contradictions:
            n_flagged += 1
            for c in contradictions:
                print(f"{row['protein_a']}/{row['protein_b']} [{row.get('verdict')}]: {c}")
        results.append({"protein_a": row["protein_a"], "protein_b": row["protein_b"],
                         "original_verdict": row.get("verdict"), "facts": facts, "contradictions": contradictions})

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n{n_flagged}/{len(rows)} candidates have a genuine evidence contradiction flagged. Wrote {args.out}")


if __name__ == "__main__":
    main()
