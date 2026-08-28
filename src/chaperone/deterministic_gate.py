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
import re
import sys
from pathlib import Path


from .sources.uniprot_client import fetch_uniprot_annotation  # noqa: E402
from .paths import PROJECT_ROOT as ROOT  # noqa: E402

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
            intracellular_only = bool(any("intracellular" in c for c in classes) and not has_any_membrane_presence)
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
    return {"status": "pass", "trace": "Real fold executed; no PTM/glycosylation site at the interface and no membrane-topology violation."}


def compute_gate_facts(row: dict, tool_calls: list, fold_results: list) -> dict:
    gene_a, gene_b = row["protein_a"], row["protein_b"]
    known = known_interaction_strength(tool_calls)
    hpa = hpa_plausibility(tool_calls, gene_a, gene_b)
    fold = fold_evidence_status(fold_results)
    glycan_binding_gene = _glycan_binding_gene(gene_a, gene_b)
    return {"known_interaction_strength": known["strength"], "known_interaction_trace": known["trace"],
            "hpa_plausibility": hpa["plausibility"], "hpa_trace": hpa["trace"],
            "fold_evidence_status": fold["status"], "fold_trace": fold["trace"],
            "glycan_binding_gene": glycan_binding_gene}


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

    if facts["hpa_plausibility"] == "implausible" and verdict != "IMPLAUSIBLE":
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
    if facts["fold_evidence_status"] == "pass" and verdict == "LIKELY_ARTIFACT_PTM" and not facts.get("glycan_binding_gene"):
        contradictions.append(f"A real AF3 fold was executed specifically to check the PTM/topology concern behind this LIKELY_ARTIFACT_PTM verdict, and found no problem — the concern this verdict is based on has been directly tested and not confirmed. {facts['fold_trace']}")

    return contradictions


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
    results = list(fold_by_pair.get(f"{row['protein_a']}/{row['protein_b']}", []))
    for call in tool_calls:
        if call.get("tool") == "fold_complex" and isinstance(call.get("output"), dict) and "iptm" in call["output"]:
            results.append(call["output"])
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
